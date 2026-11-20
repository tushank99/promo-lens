import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promolense import pipeline  # noqa: E402
from promolense.llm import load_config  # noqa: E402
from promolense.parser import parse_file  # noqa: E402

EP1 = ROOT / "episodes" / "episode-001-Episode 1671703.txt"
CFG = load_config()
SENTINEL = "SENTINEL-PROPOSER-RATIONALE-XYZ"


def good_judge(first, last, **over):
    """A strong judgement citing real unit ids from the clip."""
    j = {"hook": {"score": 4, "evidence_unit_ids": [first]},
         "clarity": {"score": 4, "evidence_unit_ids": [first]},
         "emotion": {"score": 4, "evidence_unit_ids": [first]},
         "escalation": {"score": 3, "evidence_unit_ids": [last]},
         "ending": {"score": 4, "type": "cliffhanger", "evidence_unit_ids": [last]},
         "hook_unit_id": first, "hook_type": "reveal", "emotion_type": "shock", "trope": "betrayed bride",
         "central_conflict": "A bride is betrayed.", "spoils_main_twist": False}
    j.update(over)
    return j


def make_llm(proposals, judge_for=None):
    """judge_for(first_id, last_id, clip_ids, prompt) -> dict (default: good_judge)."""
    calls = {"propose": 0, "judge": 0, "judge_prompts": []}

    def llm(stage, prompt, system, use_cache, meta):
        meta.update(provider="fake", model="fake", cached=False)
        calls[stage] += 1
        if stage == "propose":
            return {"candidates": proposals}
        calls["judge_prompts"].append(prompt)
        ids = re.findall(r"^(e01_u\d{3}) \|", prompt, re.M)
        f = (judge_for or (lambda a, b, c, p: good_judge(a, b)))
        return f(ids[0], ids[-1], ids, prompt)
    return llm, calls


def P(s, e, h, cl=None, reason=SENTINEL):
    return {"start_unit_id": s, "end_unit_id": e, "hook_unit_id": h, "cliffhanger_unit_id": cl, "reason": reason}


H2 = P("e01_u031", "e01_u048", "e01_u032", "e01_u048")   # 46.8-72.4, passes validator
H3 = P("e01_u069", "e01_u095", "e01_u069", "e01_u094")   # 98.8-138.0
H1 = P("e01_u021", "e01_u029", "e01_u021", "e01_u029")   # 11.1 s: too short
H2_DUP = P("e01_u031", "e01_u045", "e01_u032", None)     # shorter cut (46.8-68.8) of the same moment
H3_ANGLE = P("e01_u081", "e01_u095", "e01_u082", "e01_u094")  # overlaps H3, different hook (117.0 vs 98.8)


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = parse_file(EP1)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_pipe(self, proposals, judge_for=None):
        llm, calls = make_llm(proposals, judge_for)
        final = pipeline.run_episode(self.ep, self.tmp, CFG, llm=llm)
        return final, calls

    def by_id(self, final):
        return {c["candidate_id"]: c for c in final["all_candidates"]}

    # ---- trail + validation
    def test_trail_files_and_rejection_reasons(self):
        bad_id = P("e01_u999", "e01_u048", "e01_u032")
        malformed = {"start_unit_id": "e01_u031"}
        final, _ = self.run_pipe([H2, H3, H1, bad_id, malformed])
        for f in ("proposals_raw.json", "validation.json", "judged.json", "final.json", "final.md"):
            self.assertTrue((self.tmp / f).exists(), f)
        self.assertTrue((self.tmp / "prompts" / "propose.txt").exists())
        c = self.by_id(final)
        self.assertEqual(c["c03"]["rejection_reasons"], ["validation:duration"])
        self.assertEqual(c["c04"]["rejection_reasons"], ["validation:ids"])
        self.assertEqual(c["c05"]["rejection_reasons"], ["validation:malformed_proposal"])
        self.assertEqual(c["c03"]["rejected_at"], "validation")
        self.assertEqual({k["candidate_id"] for k in final["final"]}, {"c01", "c02"})
        self.assertEqual(final["counts"]["proposed"], 5)
        self.assertEqual(final["counts"]["rejected_at_validation"], 3)

    def test_required_fields_on_passed_candidate(self):
        final, _ = self.run_pipe([H2])
        k = final["final"][0]
        for f in ("candidate_id", "start_unit_id", "end_unit_id", "hook_unit_id", "hook_type", "emotion_type",
                  "scores", "evidence_unit_ids", "spoils_main_twist", "rejection_reasons", "final_score", "segments"):
            self.assertIn(f, k)
        self.assertEqual((k["start"], k["end"]), (46.8, 72.4))
        self.assertEqual(k["segments"][0]["start_unit"], "e01_u031")  # original window preserved

    def test_zero_candidates_is_fine(self):
        final, calls = self.run_pipe([])
        self.assertEqual(final["final"], [])
        self.assertEqual(calls["judge"], 0)
        self.assertIn("No promo passed", (self.tmp / "final.md").read_text(encoding="utf-8"))

    # ---- judge independence
    def test_judge_never_sees_proposer_rationale_or_hook_claim(self):
        _, calls = self.run_pipe([P("e01_u031", "e01_u048", "e01_u032", "e01_u048", reason=SENTINEL)])
        self.assertTrue(calls["judge_prompts"])
        for pr in calls["judge_prompts"]:
            self.assertNotIn(SENTINEL, pr)
            self.assertNotIn("hook_unit_id: e01", pr)
            self.assertNotIn("e01_u030 |", pr)  # unit outside the clip window is not shown

    # ---- floors and evidence
    def test_clarity_floor_rejects_and_keeps_scores(self):
        final, _ = self.run_pipe([H2], lambda a, b, ids, p: good_judge(a, b, clarity={"score": 2, "evidence_unit_ids": [a]}))
        c = self.by_id(final)["c01"]
        self.assertEqual(c["status"], "rejected")
        self.assertEqual(c["rejected_at"], "floors")
        self.assertEqual(c["rejection_reasons"], ["floor:clarity<3"])
        self.assertEqual(c["scores"]["clarity"], 2)

    def test_hook_floor(self):
        final, _ = self.run_pipe([H2], lambda a, b, ids, p: good_judge(a, b, hook={"score": 1, "evidence_unit_ids": [a]}))
        self.assertIn("floor:hook<3", self.by_id(final)["c01"]["rejection_reasons"])

    def test_no_floor_on_ending(self):
        final, _ = self.run_pipe([H2], lambda a, b, ids, p: good_judge(a, b, ending={"score": 0, "type": "neither", "evidence_unit_ids": []}))
        self.assertEqual(final["final"][0]["candidate_id"], "c01")  # still passes: ending is ranking only

    def test_evidence_outside_clip_is_dropped_and_gate_needs_valid_evidence(self):
        final, _ = self.run_pipe([H2], lambda a, b, ids, p: good_judge(
            a, b, hook={"score": 5, "evidence_unit_ids": ["e01_u090"]}))  # u090 is not in the clip
        c = self.by_id(final)["c01"]
        self.assertIn("judge_cited_ids_outside_clip", c["flags"])
        self.assertEqual(c["evidence_unit_ids"]["hook"], [])
        self.assertEqual(c["rejection_reasons"], ["unsupported:hook_has_no_valid_evidence"])

    def test_early_ending_evidence_is_penalised_not_rejected(self):
        final, _ = self.run_pipe([H2], lambda a, b, ids, p: good_judge(
            a, b, ending={"score": 5, "type": "cliffhanger", "evidence_unit_ids": [a]}))
        k = final["final"][0]
        self.assertIn("ending_evidence_not_in_final_20pct", k["flags"])
        self.assertEqual(k["scores"]["ending"], 1)

    def test_invalid_judge_output_is_recorded_not_dropped(self):
        final, _ = self.run_pipe([H2, H3], lambda a, b, ids, p: {"hook": 4} if a == "e01_u031" else good_judge(a, b))
        c = self.by_id(final)
        self.assertEqual(c["c01"]["rejection_reasons"], ["judge:invalid_output"])
        self.assertEqual(c["c02"]["status"], "final")

    def test_judge_call_failure_is_recorded(self):
        def boom(a, b, ids, p):
            raise RuntimeError("quota")
        final, _ = self.run_pipe([H2], boom)
        c = self.by_id(final)["c01"]
        self.assertEqual(c["rejection_reasons"], ["judge:failed"])
        self.assertIn("quota", c["rejection_details"][0])

    # ---- ranking, spoilers, dedupe
    def test_ranking_by_score(self):
        def jf(a, b, ids, p):
            return good_judge(a, b, emotion={"score": 5 if a == "e01_u069" else 3, "evidence_unit_ids": [a]})
        final, _ = self.run_pipe([H2, H3], jf)
        self.assertEqual([k["candidate_id"] for k in final["final"]], ["c02", "c01"])
        self.assertEqual([k["rank"] for k in final["final"]], [1, 2])

    def test_spoiler_is_demoted_not_deleted(self):
        final, _ = self.run_pipe([H2, H3], lambda a, b, ids, p: good_judge(a, b, spoils_main_twist=(a == "e01_u031")))
        ids = [k["candidate_id"] for k in final["final"]]
        self.assertEqual(ids, ["c02", "c01"])
        c1 = self.by_id(final)["c01"]
        self.assertIn("spoils_main_twist", c1["flags"])
        self.assertAlmostEqual(c1["final_score"], round(self.by_id(final)["c02"]["final_score"] * 0.85, 1), 1)

    def test_dedupe_same_hook_becomes_alternate(self):
        final, _ = self.run_pipe([H2, H2_DUP])
        c = self.by_id(final)
        self.assertEqual([k["candidate_id"] for k in final["final"]], ["c01"])
        self.assertEqual(c["c02"]["status"], "alternate")
        self.assertEqual(c["c02"]["alternate_of"], "c01")
        self.assertEqual(final["final"][0]["alternates"], ["c02"])

    def test_dedupe_keeps_overlapping_clip_with_different_hook(self):
        final, _ = self.run_pipe([H3, H3_ANGLE])
        ids = [k["candidate_id"] for k in final["final"]]
        self.assertEqual(sorted(ids), ["c01", "c02"])
        angle = self.by_id(final)["c02"]
        self.assertTrue(any(f.startswith("overlaps_but_different_hook") for f in angle["flags"]))

    def test_pipeline_does_not_read_human_labels(self):
        src = (ROOT / "src" / "promolense" / "pipeline.py").read_text(encoding="utf-8")
        for banned in ("_human", "ep01_human", "eval/", "eval\\", "labels"):
            self.assertNotIn(banned, src)

    def test_cached_stage_provenance_recorded(self):
        self.run_pipe([H2])
        raw = json.loads((self.tmp / "proposals_raw.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["llm"]["provider"], "fake")


if __name__ == "__main__":
    unittest.main()
