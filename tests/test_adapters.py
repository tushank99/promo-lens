import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promolense import pipeline  # noqa: E402
from promolense.adapters import build_manifest, detect_format, load_episode, parse_subtitles  # noqa: E402
from promolense.llm import load_config  # noqa: E402
from promolense.schema import from_canonical, to_canonical  # noqa: E402
from promolense.validator import validate  # noqa: E402

EP_DIR = ROOT / "episodes"
EP1 = EP_DIR / "episode-001-Episode 1671703.txt"

SRT = """1
00:00:01,000 --> 00:00:03,500
ANNA: You promised me
you would come back!

2
00:00:04,000 --> 00:00:06,000
<i>I never promised that.</i>
3
00:00:14,000 --> 00:00:16,500
- How long?

4
00:00:17,000 --> 00:00:2x,000
broken timestamp
"""

VTT = """WEBVTT

intro
00:00.500 --> 00:02.000 align:start
<v Mara>The door was open all night.</v>

00:02.200 --> 00:04.000
<v.loud Tom>Then someone was here.
"""


class SubtitleTests(unittest.TestCase):
    def test_srt(self):
        ep = parse_subtitles(SRT, number=7)
        self.assertEqual(ep.visual_coverage, "dialogue_only")
        self.assertEqual(len(ep.scenes), 1)
        self.assertEqual(len(ep.units), 4)
        u1, u2, u3, u4 = ep.units
        self.assertEqual((u1.start, u1.end, u1.speaker), (1.0, 3.5, "ANNA"))
        self.assertEqual(u1.text, "You promised me you would come back!")
        self.assertEqual(u2.text, "I never promised that.")      # tags stripped, missing blank line handled
        self.assertEqual(u3.start, 14.0)
        self.assertIn("bad_timestamp", u4.flags)
        self.assertTrue(all(u.type == "dialogue" and u.source_type == "subtitle" for u in ep.units))
        self.assertTrue(any("no camera/action units" in i for i in ep.issues))
        self.assertEqual(u1.id, "e07_u001")

    def test_vtt_voices_and_settings(self):
        ep = parse_subtitles(VTT, number=3)
        self.assertEqual([(u.speaker, u.start, u.end) for u in ep.units],
                         [("Mara", 0.5, 2.0), ("Tom", 2.2, 4.0)])
        self.assertEqual(ep.units[0].text, "The door was open all night.")

    def test_subtitle_gaps_are_unobserved_not_review(self):
        lines = "\n".join(f"{i + 1}\n00:00:{2 * i:02d},000 --> 00:00:{2 * i + 1:02d},500\nLine {i}.\n"
                          for i in range(6))
        lines += "7\n00:00:20,000 --> 00:00:22,000\nLate line.\n"   # last cue ends 11.5 -> 8.5 s gap
        ep = parse_subtitles(lines, number=1)
        r = validate(ep, {"start_unit": ep.units[0].id, "end_unit": ep.units[-1].id, "hook_unit": ep.units[0].id})
        self.assertEqual(r.status, "pass", r.review_details)
        self.assertTrue(any(w.startswith("unobserved gap 8.5") for w in r.warnings), r.warnings)


class DetectTests(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(detect_format(Path("a.srt"), ""), "subtitles")
        self.assertEqual(detect_format(Path("a.vtt"), ""), "subtitles")
        self.assertEqual(detect_format(Path("a.json"), ""), "canonical_json")
        self.assertEqual(detect_format(Path("a.txt"), "WEBVTT\n\n"), "subtitles")
        self.assertEqual(detect_format(Path("a.txt"), "1\n00:00:01,000 --> 00:00:02,000\nhi"), "subtitles")
        self.assertEqual(detect_format(Path("a.txt"), "EPISODE NUMBER: Episode 1"), "screenplay_txt")


@unittest.skipUnless(EP1.exists(), "episodes not present")
class CanonicalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_txt_to_canonical_roundtrip_is_lossless_for_the_core(self):
        ep = load_episode(EP1)
        d = to_canonical(ep)
        self.assertEqual(d["episode_id"], "ep01")
        self.assertEqual(d["source"]["format"], "screenplay_txt")
        f = self.tmp / "ep01.json"
        f.write_text(json.dumps(d), encoding="utf-8")
        ep2 = load_episode(f)
        key = lambda u: (u.id, u.scene, u.type, u.speaker, u.tone, u.start, u.end, u.text, u.exchange, u.modifiers)
        self.assertEqual([key(u) for u in ep.units], [key(u) for u in ep2.units])
        self.assertEqual((ep2.logline, ep2.number, ep2.visual_coverage), (ep.logline, 1, "full"))
        c = {"start_unit": "e01_u031", "end_unit": "e01_u048", "hook_unit": "e01_u032"}
        r1, r2 = validate(ep, c), validate(ep2, c)
        self.assertEqual((r1.status, r1.start, r1.end), (r2.status, r2.start, r2.end))

    def test_canonical_json_missing_fields_flagged_and_rejected(self):
        d = {"episode_id": "x01", "episode_number": 1, "scenes": [{"scene_id": "01", "units": [
            {"unit_id": "a", "start": 0, "end": 2, "unit_type": "dialogue", "text": "Hi"},
            {"unit_id": "b", "start": 2, "unit_type": "dialogue", "text": "No end"},
            {"unit_id": "c", "start": 3, "end": 4, "unit_type": "weird", "text": "Bad type"},
            {"start": 4, "end": 5, "unit_type": "action", "text": "No id"}]}]}
        ep = from_canonical(d)
        u = {x.id: x for x in ep.units}
        self.assertIn("missing_timestamp", u["b"].flags)
        self.assertIn("bad_type", u["c"].flags)
        self.assertTrue(any("missing_unit_id" in x.flags for x in ep.units))
        self.assertTrue(ep.rejected)

    def test_invalid_json_file_is_rejected_not_crash(self):
        f = self.tmp / "bad.json"
        f.write_text("{not json", encoding="utf-8")
        ep = load_episode(f)
        self.assertTrue(ep.rejected)
        self.assertTrue(any("invalid canonical JSON" in i for i in ep.issues))

    def test_v_o_modifier_kept(self):
        ep = load_episode(next(EP_DIR.glob("episode-005-*.txt")))  # header Episode 4: has voice-over (V.O.) lines
        vo = [u for u in ep.units if "V.O." in u.modifiers]
        self.assertTrue(vo)
        self.assertFalse(ep.rejected)

    def test_pipeline_runs_on_subtitles(self):
        f = self.tmp / "ep07.srt"
        f.write_text("\n".join(f"{i + 1}\n00:00:{2 * i:02d},000 --> 00:00:{2 * i + 1:02d},800\nANNA: Line {i}.\n"
                               for i in range(20)), encoding="utf-8")
        ep = load_episode(f)
        ids = [u.id for u in ep.units]
        prop = [{"start_unit_id": ids[0], "end_unit_id": ids[12], "hook_unit_id": ids[0],
                 "cliffhanger_unit_id": ids[12], "reason": "x"}]

        def llm(stage, prompt, system, use_cache, meta):
            if stage == "propose":
                self.assertIn("only dialogue is known", prompt)
                return {"candidates": prop}
            return {d: {"score": 4, "evidence_unit_ids": [ids[0] if d != "ending" else ids[12]]}
                    for d in pipeline.DIMENSIONS} | {"hook_unit_id": ids[0]}
        final = pipeline.run_episode(ep, self.tmp / "out", load_config(), llm=llm)
        self.assertEqual(final["counts"]["final"], 1)


@unittest.skipUnless(EP1.exists(), "episodes not present")
class ManifestTests(unittest.TestCase):
    def test_real_pack(self):
        rows = build_manifest(sorted(EP_DIR.glob("*.txt")))
        self.assertEqual(len(rows), 50)
        status = {r["source_file"]: r["status"] for r in rows}
        dups = {k: v for k, v in status.items() if v.startswith("duplicate_of")}
        self.assertEqual(list(dups), ["episode-002-Episode 1671704 (1).txt"])
        self.assertEqual(dups["episode-002-Episode 1671704 (1).txt"], "duplicate_of:episode-001-Episode 1671703.txt")
        canon = [r for r in rows if r["status"] == "canonical"]
        self.assertEqual(sorted(r["header_episode"] for r in canon), list(range(1, 50)))
        self.assertFalse([r for r in rows if r["status"].startswith(("conflict", "rejected", "error"))])

    def test_conflict_same_header_different_content(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = EP1.read_text(encoding="utf-8")
            (tmp / "episode-001-a.txt").write_text(src, encoding="utf-8")
            (tmp / "episode-002-b.txt").write_text(src.replace("Episode", "Episode ", 1).replace(" the ", " a ", 1), encoding="utf-8")
            rows = build_manifest(sorted(tmp.glob("*.txt")))
            self.assertEqual(rows[0]["status"], "canonical")
            self.assertTrue(rows[1]["status"].startswith("conflict:same_header_as"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
