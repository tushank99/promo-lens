import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promolense.parser import parse_file, parse_text, parse_ts  # noqa: E402

EP1 = ROOT / "episodes" / "episode-001-Episode 1671703.txt"

HEAD = "EPISODE NUMBER: Episode 9\nEPISODE LOGLINE: A test.\nCONTINUITY NOTES: none\n\n--- SCRIPT START ---\n\n"
SCENE = "SCENE 01\nScene heading: INT. ROOM - DAY\nScene start: 00:00.00\nScene end: 00:40.00\n\nActions and dialogue:\n\n"


def unit(label, start, end, text, raw_time=None):
    t = raw_time if raw_time is not None else f"Start: ({start}) | End: ({end})\n"
    return f"**{label}:**\n{t}{text}\n\n"


def script(*units, head=HEAD, scene=SCENE):
    return head + scene + "".join(units) + "--- SCRIPT END ---\n"


class TimestampTests(unittest.TestCase):
    def test_variants(self):
        cases = {"1:02": 62, "01:02": 62, "1:02.4": 62.4, "00:01:02": 62,
                 "01:02.30": 62.3, "(00:07.20)": 7.2, "01:02,5": 62.5}
        for raw, want in cases.items():
            self.assertAlmostEqual(parse_ts(raw), want, msg=raw)

    def test_malformed(self):
        for raw in ["", "abc", "1:2", "00:75.00", "12", None, "1:02:03:04"]:
            self.assertIsNone(parse_ts(raw), msg=str(raw))


def first_line_speaker(ep):
    return next(u for u in ep.units if u.type == "dialogue").speaker


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class Episode1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = parse_file(EP1)
        cls.raw = EP1.read_text(encoding="utf-8-sig")

    def test_not_rejected(self):
        self.assertFalse(self.ep.rejected, self.ep.reject_reason)

    def test_header(self):
        self.assertEqual(self.ep.number, 1)
        self.assertTrue(self.ep.logline and len(self.ep.logline) > 20)
        self.assertTrue(self.ep.continuity)

    def test_unit_count_matches_file(self):
        n = len(re.findall(r"^\*\*.+:\*\*\s*$", self.raw, re.M))
        self.assertEqual(len(self.ep.units), n)

    def test_scene(self):
        self.assertEqual(len(self.ep.scenes), 1)
        s = self.ep.scenes[0]
        self.assertEqual((s.start, s.end), (0.0, 138.0))
        self.assertIn("CATHEDRAL", s.heading)

    def test_all_timed_ordered_inside_scene(self):
        for u in self.ep.units:
            self.assertIsNotNone(u.start, u.id)
            self.assertLessEqual(u.start, u.end, u.id)
            self.assertNotIn("outside_scene_range", u.flags, u.id)
            self.assertEqual(u.scene, "01")

    def test_ids_unique_and_episode_scoped(self):
        ids = [u.id for u in self.ep.units]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(i.startswith("e01_u") for i in ids))

    def test_types_and_speakers(self):
        types = {u.type for u in self.ep.units}
        self.assertEqual(types, {"dialogue", "camera", "action", "sfx"})
        first_line = next(u for u in self.ep.units if u.type == "dialogue")
        self.assertTrue(first_line.speaker and first_line.speaker.isupper())
        self.assertEqual(first_line.tone, "Cold, Commanding")
        self.assertIn(f'"{first_line.text}"', self.raw)          # quotes stripped, text exactly as in the file
        self.assertEqual((first_line.start, first_line.end), (11.3, 13.1))

    def test_os_modifier_not_tone(self):
        u = next(u for u in self.ep.units if "O.S." in u.modifiers)
        self.assertEqual(u.speaker, first_line_speaker(self.ep))
        self.assertEqual(u.tone, "Firm")

    def test_last_unit_is_cut_to_black(self):
        last = self.ep.units[-1]
        self.assertEqual(last.type, "camera")
        self.assertEqual((last.start, last.end), (137.0, 138.0))

    def test_no_dialogue_overlaps_in_ep1(self):
        self.assertFalse([u for u in self.ep.units if u.exchange])


class SyntheticMalformedTests(unittest.TestCase):
    """Fix #5: parser must not rely on Ep 1 being clean."""

    def good_units(self, n=60):
        return [unit("CAMERA", f"00:{i:02d}.00", f"00:{i:02d}.50", f"Shot {i}.") for i in range(n)]

    def test_no_timestamps_at_all_is_rejected_not_estimated(self):
        txt = script(unit("A (Calm)", 0, 0, '"Hi."', raw_time=""), unit("B (Calm)", 0, 0, '"Yo."', raw_time=""))
        ep = parse_text(txt)
        self.assertTrue(ep.rejected)
        self.assertIn("No timestamps", ep.reject_reason)
        self.assertTrue(all(u.start is None for u in ep.units))

    def test_one_missing_timestamp_flagged_but_episode_kept_when_rare(self):
        us = self.good_units(60)
        us.insert(5, unit("A (Calm)", 0, 0, '"Hi."', raw_time=""))
        ep = parse_text(script(*us))
        bad = [u for u in ep.units if u.start is None]
        self.assertEqual(len(bad), 1)
        self.assertIn("missing_timestamp", bad[0].flags)
        self.assertFalse(ep.rejected)  # 1/61 = 1.6% < 2%

    def test_malformed_timestamp_flagged(self):
        us = self.good_units(60)
        us.insert(3, unit("A (Calm)", 0, 0, '"Hi."', raw_time="Start: (0:xx) | End: (00:05.00)\n"))
        ep = parse_text(script(*us))
        bad = [u for u in ep.units if "bad_timestamp" in u.flags]
        self.assertEqual(len(bad), 1)
        self.assertIsNone(bad[0].start)

    def test_too_many_bad_units_rejects_episode_with_message(self):
        us = self.good_units(10)
        us.insert(2, unit("A (Calm)", 0, 0, '"Hi."', raw_time="Start: (zz) | End: (zz)\n"))
        ep = parse_text(script(*us))
        self.assertTrue(ep.rejected)
        self.assertIn("malformed", ep.reject_reason)

    def test_end_before_start_flagged(self):
        us = self.good_units(60)
        us.insert(1, unit("A (Calm)", "00:09.00", "00:08.00", '"Hi."'))
        ep = parse_text(script(*us))
        self.assertTrue(any("bad_order" in u.flags for u in ep.units))

    def test_overlapping_dialogue_forms_exchange(self):
        us = [unit("A (Angry)", "00:01.00", "00:03.00", '"You liar."'),
              unit("B (Angry)", "00:02.00", "00:04.00", '"No, you!"'),
              unit("A (Calm)", "00:05.00", "00:06.00", '"Fine."')]
        ep = parse_text(script(*us), bad_fraction=1.0)
        d = [u for u in ep.units if u.type == "dialogue"]
        self.assertEqual(d[0].exchange, d[1].exchange)
        self.assertIsNotNone(d[0].exchange)
        self.assertIsNone(d[2].exchange)

    def test_camera_overlapping_dialogue_is_not_an_exchange(self):
        us = [unit("CAMERA", "00:01.00", "00:04.00", "Close-up."),
              unit("A (Angry)", "00:02.00", "00:03.00", '"Hey."')]
        ep = parse_text(script(*us))
        self.assertFalse([u for u in ep.units if u.exchange])

    def test_missing_unit_fields(self):
        us = self.good_units(60)
        us.insert(2, unit("A (Calm)", "00:01.00", "00:02.00", ""))  # no text
        us.insert(4, unit("B", "00:03.00", "00:04.00", '"No tone."'))  # no tone
        ep = parse_text(script(*us))
        no_text = [u for u in ep.units if "missing_text" in u.flags]
        no_tone = [u for u in ep.units if "missing_tone" in u.flags]
        self.assertEqual(len(no_text), 1)
        self.assertEqual(len(no_tone), 1)
        self.assertIsNone(no_tone[0].tone)
        self.assertEqual(no_tone[0].speaker, "B")

    def test_voice_over_without_tone_is_not_bad(self):
        """Regression: Ep 4 / Ep 19 were rejected because **JULIANA (V.O.):** lines (no tone) counted as bad."""
        us = self.good_units(10) + [unit("JULIANA (V.O.)", "00:20.00", "00:22.00", '"Inner voice."'),
                                    unit("INFORMANT (O.S.)", "00:22.00", "00:24.00", '"Off screen."')]
        ep = parse_text(script(*us))
        self.assertFalse(ep.rejected, ep.reject_reason)
        vo = ep.units[-2]
        self.assertEqual((vo.speaker, vo.tone, vo.flags), ("JULIANA", None, ["missing_tone"]))
        self.assertFalse(vo.bad)

    def test_missing_headers_and_scene_do_not_crash(self):
        txt = "--- SCRIPT START ---\n" + unit("A (Calm)", "00:01.00", "00:02.00", '"Hi."') + "--- SCRIPT END ---\n"
        ep = parse_text(txt, fallback_number=7)
        self.assertEqual(ep.number, 7)
        self.assertIn("missing logline", ep.issues)
        self.assertIn("no_scene", ep.units[0].flags)

    def test_unit_outside_scene_range_flagged(self):
        us = self.good_units(60)
        us.append(unit("A (Calm)", "00:50.00", "00:55.00", '"Late."'))  # scene ends 00:40
        ep = parse_text(script(*us))
        self.assertIn("outside_scene_range", ep.units[-1].flags)

    def test_empty_input(self):
        ep = parse_text("")
        self.assertTrue(ep.rejected)

    def test_timestamp_variants_inside_script(self):
        us = self.good_units(60)
        us.append(unit("A (Calm)", "0:38", "0:39.5", '"Short form."'))
        ep = parse_text(script(*us))
        u = ep.units[-1]
        self.assertEqual((u.start, u.end), (38.0, 39.5))


if __name__ == "__main__":
    unittest.main()
