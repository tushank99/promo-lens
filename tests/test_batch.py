import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promolense.batch import run_batch  # noqa: E402
from promolense.llm import load_config  # noqa: E402

EP_DIR = ROOT / "episodes"


@unittest.skipUnless(EP_DIR.exists(), "episodes not present")
class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.files = sorted(EP_DIR.glob("episode-00[1-4]-*.txt"))  # header eps 1,1(dup),2,3
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def llm(self, fail_on=None):
        def call(stage, prompt, system, use_cache, meta):
            self.calls.append(stage)
            meta.update(provider="fake", model="fake", tokens_in=10, tokens_out=5, attempts=1)
            if fail_on and fail_on in prompt:
                raise RuntimeError("provider down")
            return {"candidates": []}
        return call

    def test_isolation_resume_and_duplicate_skip(self):
        logline_ep2 = None
        for f in self.files:
            if f.read_text(encoding="utf-8").startswith("EPISODE NUMBER: Episode 2"):
                logline_ep2 = f.read_text(encoding="utf-8").split("\n")[1][18:60]
        rows = run_batch(self.files, self.tmp, load_config(), llm=self.llm(fail_on=logline_ep2))
        st = {r["episode"]: r["status"] for r in rows}
        self.assertEqual(st, {1: "complete_no_clips", 2: "failed", 3: "complete_no_clips"})   # duplicate ep 1 processed once
        self.assertIn("provider down", next(r for r in rows if r["episode"] == 2)["error"])
        self.assertTrue((self.tmp / "episode_002" / "error.txt").exists())
        self.assertTrue((self.tmp / "summary.csv").exists())
        man = json.loads((self.tmp / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(any(r["status"].startswith("duplicate_of") for r in man))
        # resume: only the failed episode is rerun
        self.calls.clear()
        rows = run_batch(self.files, self.tmp, load_config(), llm=self.llm())
        self.assertEqual({r["episode"]: r["status"] for r in rows}, {1: "complete_no_clips", 2: "complete_no_clips", 3: "complete_no_clips"})
        self.assertEqual(self.calls, ["propose"])  # one call: episode 2 only

    def test_partial_rerun_keeps_other_episodes_in_summary(self):
        run_batch(self.files, self.tmp, load_config(), llm=self.llm())
        run_batch(self.files, self.tmp, load_config(), episodes=[3], llm=self.llm(), force=True)
        summ = json.loads((self.tmp / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual([r["episode"] for r in summ], [1, 2, 3])
        self.assertIn("episode,config_id,status", (self.tmp / "summary.csv").read_text(encoding="utf-8"))

    def test_frozen_config_refuses_change_unless_named_continuation(self):
        import copy
        from promolense.batch import ConfigChanged
        cfg = load_config()
        run_batch(self.files, self.tmp, cfg, episodes=[1], llm=self.llm())
        frozen = (self.tmp / "run_config.json").read_text(encoding="utf-8")
        cfg2 = copy.deepcopy(cfg)
        cfg2["stages"]["propose"]["model"] = "gemini-3-flash-preview"          # quota contingency switch
        with self.assertRaises(ConfigChanged):
            run_batch(self.files, self.tmp, cfg2, episodes=[2], llm=self.llm())
        rows = run_batch(self.files, self.tmp, cfg2, episodes=[2], llm=self.llm(), continuation="flash-preview")
        self.assertEqual(rows[0]["config_id"], "continuation-flash-preview")
        self.assertEqual((self.tmp / "run_config.json").read_text(encoding="utf-8"), frozen)   # immutable
        cont = json.loads((self.tmp / "run_config.continuation-flash-preview.json").read_text(encoding="utf-8"))
        self.assertIn("models", cont["changes_vs_previous"])
        summ = {r["episode"]: r["config_id"] for r in json.loads((self.tmp / "summary.json").read_text(encoding="utf-8"))}
        self.assertEqual(summ, {1: "base", 2: "continuation-flash-preview"})

    def test_quota_exhaustion_marks_pending_not_failed(self):
        def call(stage, prompt, system, use_cache, meta):
            raise RuntimeError("quota exhausted for gemini/gemini-3.1-flash-lite (not retried): daily")
        rows = run_batch(self.files, self.tmp, load_config(), llm=call)
        self.assertEqual({r["status"] for r in rows}, {"pending_quota"})
        rows = run_batch(self.files, self.tmp, load_config(), llm=self.llm())            # resume picks them up
        self.assertEqual({r["status"] for r in rows}, {"complete_no_clips"})

    def test_qa_report(self):
        from promolense.qa import run_qa
        run_batch(self.files, self.tmp, load_config(), llm=self.llm())
        rep = run_qa(self.tmp)
        self.assertTrue(rep["passed"], rep["problems"])
        self.assertTrue(rep["complete"])
        self.assertEqual(rep["checks"]["canonical_episodes"], 3)
        self.assertEqual(len(rep["checks"]["duplicates_skipped"]), 1)

    def test_rows_from_before_the_freeze_are_rerun_not_skipped(self):
        run_batch(self.files, self.tmp, load_config(), episodes=[1], llm=self.llm())
        summ = json.loads((self.tmp / "summary.json").read_text(encoding="utf-8"))
        summ[0]["config_id"] = ""                                           # legacy row (pre-freeze)
        (self.tmp / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
        rows = run_batch(self.files, self.tmp, load_config(), episodes=[1], llm=self.llm())
        self.assertNotIn("note", rows[0])                                   # re-run, not skipped
        self.assertEqual(rows[0]["config_id"], "base")
        rows = run_batch(self.files, self.tmp, load_config(), episodes=[1], llm=self.llm())
        self.assertEqual(rows[0]["note"], "skipped: already complete")

    def test_select_specific_episodes(self):
        rows = run_batch(self.files, self.tmp, load_config(), episodes=[3], llm=self.llm())
        self.assertEqual([r["episode"] for r in rows], [3])


if __name__ == "__main__":
    unittest.main()
