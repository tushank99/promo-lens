import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promolense import pipeline  # noqa: E402
from promolense.evaluate import evaluate_run, write_review_sheet  # noqa: E402
from promolense.llm import load_config  # noqa: E402
from promolense.parser import parse_file  # noqa: E402
from test_pipeline import H1, H2, H3, H3_ANGLE, make_llm  # noqa: E402

EP1 = ROOT / "episodes" / "episode-001-Episode 1671703.txt"
LABELS = ROOT / "eval" / "ep01_human.json"
CFG = load_config()


@unittest.skipUnless(EP1.exists() and LABELS.exists(), "Ep 1 / labels not present")
class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = parse_file(EP1)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_pipe(self, proposals, sub="r"):
        llm, _ = make_llm(proposals)
        pipeline.run_episode(self.ep, self.tmp / sub, CFG, llm=llm)
        return self.tmp / sub

    def test_exact_labels_recovered_with_exact_boundaries(self):
        res = evaluate_run(self.ep, self.run_pipe([H2, H3, H1]), LABELS)
        rows = {r["label"]: r for r in res["labels"]}
        self.assertTrue(rows["h2"]["final_recovered"])
        self.assertTrue(rows["h2"]["boundary_within_1_unit"])
        self.assertEqual(rows["h2"]["best_match"]["iou"], 1.0)
        # h1 is out of policy: discovered and proposed, then lost to the duration gate
        self.assertTrue(rows["h1"]["discovered"])
        self.assertFalse(rows["h1"]["final_recovered"])
        self.assertEqual(rows["h1"]["rejection_loss"][0]["reasons"], ["validation:duration"])
        self.assertIn("policy_note", rows["h1"])
        self.assertEqual(res["summary"]["final_recovery"], "2/2 in-policy labels")

    def test_openings_and_coverage(self):
        res = evaluate_run(self.ep, self.run_pipe([H2, H3]), LABELS)
        o = res["openings"]["c01"]  # H2 starts at 46.8 on a camera shot + simultaneous dialogue
        self.assertEqual(o["hook_delay_s"], 0.0)
        self.assertEqual(o["silent_opening_s"], 0.0)
        self.assertEqual(o["opening_type"], "dialogue")
        cov = res["time_coverage"]["final"]
        self.assertEqual(cov["first_third"]["share_covered"], 0.0)
        self.assertEqual(cov["last_third"]["share_covered"], 0.85)  # 98.8-138 of 92-138

    def test_overlapping_finals_reported(self):
        res = evaluate_run(self.ep, self.run_pipe([H3, H3_ANGLE]), LABELS)
        self.assertTrue(res["final_angle_proxies"]["overlapping_final_pairs"])

    def test_stability_identical_runs(self):
        a = self.run_pipe([H2, H3], "a")
        b = self.run_pipe([H3, H2], "b")
        s = evaluate_run(self.ep, a, LABELS, compare_dir=b)["stability"]
        self.assertEqual(s["final_a_found_in_b"], "2/2")
        self.assertEqual(s["final_b_found_in_a"], "2/2")

    def test_review_sheet_preserves_ratings(self):
        run = self.run_pipe([H2, H3])
        sheet = self.tmp / "review.csv"
        self.assertEqual(write_review_sheet(self.ep, [run], sheet), 2)
        txt = sheet.read_text(encoding="utf-8").replace(",,,,\n", ",y,4,sister rivalry,payoff,\n", 1)
        sheet.write_text(txt, encoding="utf-8")
        write_review_sheet(self.ep, [run], sheet)
        self.assertIn("sister rivalry", sheet.read_text(encoding="utf-8"))

    def test_timeline_facts_list_real_gaps(self):
        t = pipeline.timeline_facts(self.ep, CFG["rules"])
        self.assertIn("NOTHING scripted over 3 s (allowed, flagged for human review): none", t)
        self.assertIn("Camera-only stretches (visual storytelling, no dialogue/action; fully allowed): "
                      "29.8-33.8 s; 43.1-46.8 s", t)
        self.assertIn("exchanges (include all members or none): none", t)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class HumanAndDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_review_summary_calibration(self):
        import csv
        from promolense.evaluate import review_summary
        from test_pipeline import good_judge
        ep = parse_file(EP1)

        def jf(a, b, ids, p):  # H3 scores higher than H2
            return good_judge(a, b, emotion={"score": 5 if a == "e01_u069" else 2, "evidence_unit_ids": [a]})
        llm, _ = make_llm([H2, H3], jf)
        pipeline.run_episode(ep, self.tmp / "runX", CFG, llm=llm)
        sheet = self.tmp / "s.csv"
        write_review_sheet(ep, [self.tmp / "runX"], sheet)
        with open(sheet, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:  # human agrees with the judge: publishes the higher-scored one only
            r.update({"would_publish(y/n)": "y" if r["candidate_id"] == "c02" else "n", "opening_strength(1-5)": "4",
                      "promo_angle": "family_rejection", "ending_type_human": "cliffhanger"})
        with open(sheet, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res = review_summary(sheet, self.tmp)
        self.assertEqual(res["rated"], 2)
        self.assertEqual(res["publish_rate"], 0.5)
        self.assertEqual(res["judge_pairwise_win_rate"], 1.0)
        self.assertEqual(res["ending_type_agreement"], "2/2")
        self.assertEqual(res["angles_per_run"]["runX"]["distinct"], 1)

    def test_discovery_counts_valid_cover_only(self):
        from promolense.batch import run_batch
        from promolense.evaluate import discovery
        files = [EP1]
        llm_a, _ = make_llm([H2])                  # covers moment fully, valid
        llm_b, _ = make_llm([H1])                  # covers it? no: H1 is a different moment and too short
        run_batch(files, self.tmp / "a", CFG, llm=llm_a)
        run_batch(files, self.tmp / "b", CFG, llm=llm_b)
        m = self.tmp / "m.csv"
        m.write_text("episode,moment_name,start_unit,end_unit,why_compelling\n1,h2,e01_u031,e01_u048,x\n"
                     "1,h1,e01_u021,e01_u029,x\n", encoding="utf-8")
        res = {x["moment"]: x for x in discovery(m, [self.tmp / "a", self.tmp / "b"])["moments"]}
        self.assertEqual(res["h2"]["discovered_runs"], "1/2")
        self.assertEqual(res["h1"]["discovered_runs"], "0/2")   # run b proposed it but it was rejected (duration)
        self.assertIn("better cover by rejected", res["h1"]["detail"][1])
        self.assertIn("validation:duration", res["h1"]["detail"][1])
