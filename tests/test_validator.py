import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promolense.parser import parse_file, parse_text  # noqa: E402
from promolense.validator import validate  # noqa: E402

EP1 = ROOT / "episodes" / "episode-001-Episode 1671703.txt"
LABELS = ROOT / "eval" / "ep01_human.json"


# ---------- synthetic episode builder ----------
def build(scenes, number=9):
    """scenes: {scene_id: [(label, start, end, text), ...]} with times in seconds."""
    def ts(x):
        return f"{int(x // 60):02d}:{x % 60:05.2f}"
    out = f"EPISODE NUMBER: Episode {number}\nEPISODE LOGLINE: x\nCONTINUITY NOTES: x\n\n--- SCRIPT START ---\n\n"
    for sid, us in scenes.items():
        lo = min(u[1] for u in us); hi = max(u[2] for u in us)
        out += f"SCENE {sid}\nScene heading: INT. X\nScene start: {ts(lo)}\nScene end: {ts(hi)}\n\nActions and dialogue:\n\n"
        for label, a, b, text in us:
            out += f"**{label}:**\nStart: ({ts(a)}) | End: ({ts(b)})\n{text}\n\n"
    out += "--- SCRIPT END ---\n"
    return parse_text(out)


def lines(n, step, length, t0=0.0, who="A (Angry)"):
    return [(who, t0 + i * step, t0 + i * step + length, f'"Line {i}."') for i in range(n)]


def cand(ep, s, e, hook=None, cliff=None, **kw):
    return {"start_unit": s, "end_unit": e, "hook_unit": hook or s, "cliffhanger_unit": cliff, **kw}


@unittest.skipUnless(EP1.exists() and LABELS.exists(), "Ep 1 / labels not present")
class HumanLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = parse_file(EP1)
        cls.labels = {l["id"]: l for l in json.loads(LABELS.read_text(encoding="utf-8"))["labels"]}

    def run_label(self, lid, **over):
        l = self.labels[lid]
        c = {"start_unit": l["start_unit"], "end_unit": l["end_unit"],
             "hook_unit": l["hook_unit"], "cliffhanger_unit": l["cliffhanger_unit"], **over}
        return validate(self.ep, c)

    def test_h1_rejected_for_duration_only(self):
        r = self.run_label("h1")
        self.assertFalse(r.ok)
        self.assertEqual(r.reasons, ["duration"], r.details)
        self.assertEqual((r.start, r.end, r.duration), (32.8, 43.9, 11.1))

    def test_h2_passes_and_ends_at_72_4_not_72_1(self):
        r = self.run_label("h2")
        self.assertTrue(r.ok, r.details)
        self.assertEqual(r.end, 72.4)
        self.assertNotEqual(r.end, 72.1)
        self.assertEqual(r.duration, 25.6)

    def test_h3_passes(self):
        r = self.run_label("h3")
        self.assertTrue(r.ok, r.details)
        self.assertEqual((r.start, r.end, r.duration), (98.8, 138.0, 39.2))

    def test_mid_line_cut_rejected(self):
        # camera unit e01_u025 starts at 36.5 s, inside a spoken line that runs 35.4-37.7 s
        cam = next(u for u in self.ep.units if u.id == "e01_u025")
        self.assertEqual((cam.type, cam.start), ("camera", 36.5))
        end = "e01_u048"
        r = validate(self.ep, cand(self.ep, cam.id, end, hook=cam.id, cliff=end))
        self.assertIn("dialogue_cut", r.reasons, r.details)
        self.assertFalse(r.ok)
        # The 3.7 s camera-only reaction beat after "No..." (43.1-46.8) is occupied timeline: noted, not flagged.
        self.assertEqual(r.review, [])
        self.assertTrue(any(w.startswith("camera-only stretch 3.7") for w in r.warnings), r.warnings)

    def test_late_hook_rejected(self):
        l = self.labels["h2"]
        r = validate(self.ep, cand(self.ep, l["start_unit"], l["end_unit"], hook="e01_u048", cliff="e01_u048"))
        self.assertEqual(r.reasons, ["hook"], r.details)

    def test_early_cliffhanger_rejected(self):
        l = self.labels["h2"]
        r = validate(self.ep, cand(self.ep, l["start_unit"], l["end_unit"], hook=l["hook_unit"], cliff=l["hook_unit"]))
        self.assertEqual(r.reasons, ["cliffhanger"], r.details)

    def test_visual_hook_is_allowed(self):
        # h1's hook unit is a camera shot (visual), must not be penalised for not being speech
        h1 = self.labels["h1"]
        u = next(u for u in self.ep.units if u.id == h1["hook_unit"])
        self.assertEqual(u.type, "camera")
        self.assertNotIn("hook", self.run_label("h1").reasons)

    def test_invalid_ids_rejected(self):
        for bad in ({"start_unit": "e01_u999"}, {"end_unit": "nope"}, {"hook_unit": "e01_u500"}):
            r = self.run_label("h2", **bad)
            self.assertEqual(r.reasons, ["ids"], r.details)

    def test_reversed_order_rejected(self):
        l = self.labels["h2"]
        r = validate(self.ep, cand(self.ep, l["end_unit"], l["start_unit"]))
        self.assertEqual(r.reasons, ["order"])

    def test_no_repair_and_no_mutation(self):
        before = copy.deepcopy(self.ep.to_dict())
        c = {"start_unit": "e01_u021", "end_unit": "e01_u029", "hook_unit": "e01_u021", "cliffhanger_unit": "e01_u029"}
        c0 = dict(c)
        r = self.run_label("h1")
        self.assertEqual(self.ep.to_dict(), before)
        self.assertEqual(c, c0)
        # reports the ORIGINAL window; never extends/shifts to reach 20 s
        self.assertEqual((r.start, r.end), (32.8, 43.9))
        self.assertEqual(r.segments[0]["start_unit"], "e01_u021")

    def test_altar_reveal_now_reachable(self):
        """Phase 5 finding: 0 valid windows contained the reveal under the 3 s hard rule. Camera units now count
        as occupied timeline, so windows around the reveal pass outright."""
        ids = [u.id for u in self.ep.units]
        a_max, b_min = ids.index("e01_u021"), ids.index("e01_u026")
        found = []
        for a in range(a_max + 1):
            for b in range(b_min, min(b_min + 40, len(ids))):
                r = validate(self.ep, {"start_unit": ids[a], "end_unit": ids[b], "hook_unit": ids[a]})
                if r.ok:
                    found.append(r.status)
        self.assertTrue(found)
        self.assertIn("pass", found)
        # the windows used by runs 1-3 for the altar moment
        r = validate(self.ep, {"start_unit": "e01_u007", "end_unit": "e01_u027", "hook_unit": "e01_u008"})
        self.assertEqual(r.status, "pass", (r.details, r.review_details))

    def test_all_three_summary(self):
        got = {k: self.run_label(k).ok for k in self.labels}
        self.assertEqual(got, {"h1": False, "h2": True, "h3": True})


class SyntheticTests(unittest.TestCase):
    def test_exactly_20_and_60_inclusive(self):
        for total in (20.0, 60.0):
            n = int(total // 2)
            ep = build({"01": lines(n, 2.0, 2.0)})  # contiguous, 2 s each -> clip = n*2 = total
            r = validate(ep, cand(ep, ep.units[0].id, ep.units[-1].id, cliff=ep.units[-1].id))
            self.assertEqual(r.duration, total)
            self.assertTrue(r.ok, (total, r.details))

    def test_just_outside_limits_rejected(self):
        for total in (19.9, 60.1):
            ep = build({"01": [("A (Angry)", 0, total, '"One long line."')]})
            r = validate(ep, cand(ep, "e09_u001", "e09_u001", cliff="e09_u001"))
            self.assertIn("duration", r.reasons, total)

    def test_long_camera_only_stretch_is_occupied_not_dead_air(self):
        us = [("CAMERA", 0, 26, "Wide shot.")] + lines(5, 3.0, 2.0) + lines(2, 3.0, 2.0, t0=20.0)
        ep = build({"01": us})  # speech 0-14, camera only 14-20 (6 s), speech 20-26
        r = validate(ep, cand(ep, ep.units[0].id, ep.units[-1].id, hook=ep.units[1].id, cliff=ep.units[-1].id))
        self.assertEqual(r.status, "pass", (r.details, r.review_details))
        self.assertTrue(any("camera-only stretch 6.0" in w for w in r.warnings), r.warnings)

    def _gap_ep(self, gap, camera=True):
        """26+ s of speech with one silent gap of `gap` seconds after 12 s; optional camera over the gap."""
        us = lines(5, 2.5, 2.5) + lines(6, 2.5, 2.5, t0=12.5 + gap)
        if camera:
            us = [("CAMERA", 0, 12.5 + gap + 15, "Long shot.")] + us
        return build({"01": us})

    def _run_gap(self, gap, camera=True):
        ep = self._gap_ep(gap, camera)
        ids = [u.id for u in ep.units]
        last_line = [u.id for u in ep.units if u.type == "dialogue"][-1]
        return validate(ep, cand(ep, ids[0], ids[-1], hook=ids[1] if camera else ids[0], cliff=last_line))

    def test_gap_policy_bands(self):
        """Camera-covered gaps of any length pass; uncovered gaps > 3 s are review, never reject."""
        for gap in (3.0, 4.0, 8.0, 15.0):
            self.assertEqual(self._run_gap(gap, camera=True).status, "pass", gap)
        for gap, want in [(3.0, "pass"), (3.5, "review"), (5.0, "review"), (8.0, "review"), (15.0, "review")]:
            r = self._run_gap(gap, camera=False)
            self.assertEqual(r.status, want, (gap, r.details, r.review_details))
            self.assertEqual(r.reasons, [])
            self.assertTrue(r.ok)
            if want == "review":
                self.assertEqual(r.review, ["uncovered_gap"])

    def test_dialogue_only_source_marks_gaps_unobserved(self):
        ep = self._gap_ep(8.0, camera=False)
        ep.visual_coverage = "dialogue_only"
        ids = [u.id for u in ep.units]
        r = validate(ep, cand(ep, ids[0], ids[-1], cliff=ids[-1]))
        self.assertEqual(r.status, "pass", r.review_details)
        self.assertTrue(any(w.startswith("unobserved gap 8.0") for w in r.warnings), r.warnings)

    def test_dense_dialogue_not_dead_air(self):
        ep = build({"01": lines(9, 3.0, 2.0)})  # 1 s gaps only; 26 s
        r = validate(ep, cand(ep, ep.units[0].id, ep.units[-1].id, cliff=ep.units[-1].id))
        self.assertTrue(r.ok, r.details)

    def test_overlapping_dialogue_accepted_and_not_silence(self):
        us = lines(4, 3.0, 2.0) + [("B (Angry)", 3.5, 6.5, '"Interrupting."')] + lines(6, 3.0, 2.0, t0=12.0)
        ep = build({"01": us})
        ex = {u.exchange for u in ep.units if u.exchange}
        self.assertTrue(ex)
        r = validate(ep, cand(ep, ep.units[0].id, ep.units[-1].id, cliff=ep.units[-1].id))
        self.assertTrue(r.ok, r.details)

    def test_split_exchange_rejected(self):
        us = lines(2, 10.0, 2.0) + [("A (Angry)", 20, 24, '"Talking."'), ("B (Angry)", 22, 28, '"Overlap."')] \
             + lines(3, 3.0, 2.0, t0=30.0)
        ep = build({"01": us})
        ids = [u.id for u in ep.units]
        # window starts at the 2nd member of the overlapping pair, so the exchange is split
        r = validate(ep, cand(ep, ids[3], ids[-1], cliff=ids[-1]))
        self.assertIn("exchange_split", r.reasons, r.details)

    def test_cross_scene_rejected_unless_allowed(self):
        ep = build({"01": lines(5, 3.0, 2.0), "02": lines(5, 3.0, 2.0, t0=15.0)})
        c = cand(ep, ep.units[0].id, ep.units[-1].id, cliff=ep.units[-1].id)
        r = validate(ep, c)
        self.assertEqual(r.reasons, ["scene"], r.details)
        self.assertTrue(validate(ep, {**c, "allow_cross_scene": True}).ok)

    def test_late_hook_boundary_3s(self):
        ep = build({"01": [("CAMERA", 0, 12, "Establishing.")] + lines(8, 3.0, 2.0, t0=0.0)})
        ids = [u.id for u in ep.units]
        ok = validate(ep, cand(ep, ids[0], ids[-1], hook=ids[2], cliff=ids[-1]))   # starts at 3.0 -> exactly 3
        late = validate(ep, cand(ep, ids[0], ids[-1], hook=ids[3], cliff=ids[-1]))  # starts at 6.0
        self.assertNotIn("hook", ok.reasons, ok.details)
        self.assertIn("hook", late.reasons)

    def test_hook_four_seconds_late_rejected(self):
        ep = build({"01": [("ACTION", 0, 4, "Quiet moment.")] + lines(9, 3.0, 2.0, t0=4.0)})
        ids = [u.id for u in ep.units]
        r = validate(ep, cand(ep, ids[0], ids[-1], hook=ids[1], cliff=ids[-1]))  # first line starts 4.0 s in
        self.assertEqual(r.reasons, ["hook"], r.details)

    def test_action_outside_window_playing_inside_is_boundary(self):
        us = [("ACTION", 0, 40, "Long action.")] + lines(10, 3.0, 2.0, t0=5.0)
        ep = build({"01": us})
        ids = [u.id for u in ep.units]
        r = validate(ep, cand(ep, ids[1], ids[-1], cliff=ids[-1]))
        self.assertEqual(r.status, "review", r.details)            # straddling action/sfx: review, not reject
        self.assertEqual(r.review, ["straddled_action_or_sfx"])

    def test_camera_straddle_is_only_a_warning(self):
        us = [("CAMERA", 0, 30, "Long shot.")] + lines(10, 3.0, 2.0, t0=5.0)
        ep = build({"01": us})
        ids = [u.id for u in ep.units]
        r = validate(ep, cand(ep, ids[1], ids[-1], cliff=ids[-1]))
        self.assertTrue(r.ok, r.details)
        self.assertTrue(r.warnings)

    def test_unparseable_timestamp_in_window_rejected(self):
        ep = build({"01": lines(9, 3.0, 2.0)})
        ep.units[4].start = None  # simulate a flagged unit surviving parse
        r = validate(ep, cand(ep, ep.units[0].id, ep.units[-1].id))
        self.assertEqual(r.reasons, ["timestamp"])

    def test_multi_segment_unsupported_in_p0(self):
        ep = build({"01": lines(9, 3.0, 2.0)})
        r = validate(ep, {"segments": [{}, {}], "start_unit": "e09_u001", "end_unit": "e09_u002", "hook_unit": "e09_u001"})
        self.assertEqual(r.reasons, ["segments"])


if __name__ == "__main__":
    unittest.main()
