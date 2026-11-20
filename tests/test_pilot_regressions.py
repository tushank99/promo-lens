"""Regression tests for the pilot_001 failure modes (hook labelling, re-proposal, judge reliability, resume).
Added alongside the older tests; none of those were relaxed."""
import io
import json
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from promolense import llm, pipeline  # noqa: E402
from promolense.adapters import load_episode  # noqa: E402
from promolense.batch import run_batch  # noqa: E402
from promolense.llm import load_config  # noqa: E402
from promolense.validator import validate  # noqa: E402
from test_pipeline import good_judge  # noqa: E402
from test_validator import build, lines  # noqa: E402

EP1 = ROOT / "episodes" / "episode-001-Episode 1671703.txt"
CFG = load_config()
# Mechanism tests below pin their own judge (Gemini primary, fallback off) so they don't depend on the frozen choice.
import copy  # noqa: E402
CFG_GJ = copy.deepcopy(CFG)
CFG_GJ["stages"]["judge"] = {"provider": "gemini", "model": "gemini-3.5-flash", "fallback": ""}


def H(ids, modality):
    return {"unit_ids": ids, "modality": modality, "description": "d", "evidence": "e"}


def P(s, e, ids, modality, cl=None, **kw):
    return {"start_unit_id": s, "end_unit_id": e, "hook": H(ids, modality), "cliffhanger_unit_id": cl,
            "reason": "r", **kw}


# ------------------------------------------------------------------ hook contract (validator)
class HookContractTests(unittest.TestCase):
    def ep(self, gap_before_speech):
        """Camera shot at 0 (the visual hook), first line `gap_before_speech` s later, then 25 s of speech."""
        us = [("CAMERA", 0.0, gap_before_speech, "Close-up: the ledger burning on the table.")]
        us += lines(9, 3.0, 2.5, t0=gap_before_speech)
        return build({"01": us})

    def check(self, ep, hook_ids, modality, s=None, e=None):
        ids = [u.id for u in ep.units]
        return validate(ep, {"start_unit": s or ids[0], "end_unit": e or ids[-1], "hook_units": hook_ids,
                             "hook_modality": modality})

    def test_visual_hook_at_start_speech_4s_later_passes(self):
        ep = self.ep(4.0)
        r = self.check(ep, [ep.units[0].id], "visual")
        self.assertEqual(r.status, "pass", r.details)
        self.assertEqual(r.hook_delay, 0.0)

    def test_visual_hook_at_start_speech_10s_later_passes(self):
        ep = self.ep(10.0)
        r = self.check(ep, [ep.units[0].id], "visual")
        self.assertEqual(r.status, "pass", r.details)

    def test_dialogue_hook_at_4s_fails(self):
        ep = self.ep(4.0)
        r = self.check(ep, [ep.units[1].id], "spoken")   # nominating the line, not the opening image
        self.assertEqual(r.reasons, ["hook"])
        self.assertEqual(r.hook_delay, 4.0)

    def test_hook_exactly_3s_passes_and_3_01_fails(self):
        for gap, ok in ((3.0, True), (3.01, False)):
            ep = self.ep(gap)
            r = self.check(ep, [ep.units[1].id], "spoken")
            self.assertEqual(r.ok, ok, (gap, r.details))

    def test_hook_unit_outside_candidate_fails(self):
        ep = self.ep(4.0)
        ids = [u.id for u in ep.units]
        r = self.check(ep, [ids[-1]], "spoken", s=ids[0], e=ids[-2])
        self.assertIn("hook", r.reasons)
        self.assertIn("outside the window", r.details[0])

    def test_modality_and_hook_ids_are_validated(self):
        ep = self.ep(4.0)
        cam, line = ep.units[0].id, ep.units[1].id
        self.assertEqual(self.check(ep, [line], "visual").reasons[-1], "hook_modality")   # visual claimed on a line
        self.assertEqual(self.check(ep, [cam], "spoken").reasons[-1], "hook_modality")    # spoken claimed on camera
        self.assertIn("hook_modality", self.check(ep, [cam], "smell").reasons)
        self.assertEqual(self.check(ep, ["e09_u999"], "visual").reasons, ["ids"])
        self.assertTrue(self.check(ep, [cam, line], "visual").ok)  # multi-unit hook: earliest counts

    def test_proposal_cannot_enable_cross_scene(self):
        ep = build({"01": lines(5, 3.0, 2.0), "02": lines(5, 3.0, 2.0, t0=15.0)})
        prop = P(ep.units[0].id, ep.units[-1].id, [ep.units[0].id], "spoken", allow_cross_scene=True)
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = pipeline.validate_all(ep, [{"candidate_id": "c01", "raw": prop}], tmp, CFG["rules"])
            self.assertEqual(rows[0]["reasons"], ["scene"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ bounded re-proposal (pipeline)
class FakeLLM:
    def __init__(self, proposals, retries=None, judge=None):
        self.proposals, self.retries, self.judge = proposals, retries, judge
        self.calls, self.prompts = [], []

    def __call__(self, stage, prompt, system, use_cache, meta):
        meta.update(provider="fake", model="fake")
        is_retry = "were rejected by the code checker" in prompt
        self.calls.append("retry" if is_retry else stage)
        self.prompts.append(prompt)
        if stage == "propose":
            return {"retries": self.retries or []} if is_retry else {"candidates": self.proposals}
        import re
        ids = re.findall(r"^(e\d\d_u\d{3}) \|", prompt, re.M)
        return (self.judge or (lambda a, b, p: good_judge(a, b)))(ids[0], ids[-1], prompt)


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class ReproposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = load_episode(EP1)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_pipe(self, fake, ep=None):
        final = pipeline.run_episode(ep or self.ep, self.tmp, CFG, llm=fake)
        return final, {c["candidate_id"]: c for c in final["all_candidates"]}

    # the real pilot failure: Ep 1 altar, spoken hook 11.3 s in
    ALTAR_ORIG = P("e01_u001", "e01_u029", ["e01_u008"], "spoken")
    ALTAR_RETRY = dict(P("e01_u007", "e01_u029", ["e01_u007"], "visual"), retry_of="c01")

    def test_hook_rejection_retry_can_change_window_and_modality(self):
        final, c = self.run_pipe(FakeLLM([self.ALTAR_ORIG], [self.ALTAR_RETRY]))
        self.assertEqual(c["c01"]["rejection_reasons"], ["validation:hook"])
        self.assertEqual(c["c01r"]["status"], "final")
        self.assertEqual(c["c01r"]["hook"]["modality"], "visual")
        self.assertEqual((c["c01r"]["start"], c["c01r"]["hook"]["delay_s"]), (11.0, 0.0))
        self.assertEqual(final["counts"]["retry_recovered"], 1)

    def test_duration_retry_can_select_new_valid_window(self):
        short = P("e01_u021", "e01_u029", ["e01_u021"], "visual")
        retry = dict(P("e01_u021", "e01_u035", ["e01_u021"], "visual"), retry_of="c01")
        _, c = self.run_pipe(FakeLLM([short], [retry]))
        self.assertEqual(c["c01"]["rejection_reasons"], ["validation:duration"])
        self.assertEqual(c["c01r"]["status"], "final")
        self.assertEqual(c["c01r"]["duration"], 21.8)

    def test_retry_cannot_bypass_scene_rule(self):
        ep = build({"01": lines(5, 3.0, 2.0), "02": lines(9, 3.0, 2.0, t0=15.0)})
        ids = [u.id for u in ep.units]
        orig = P(ids[0], ids[-1], [ids[0]], "spoken")                          # crosses scenes
        retry = dict(P(ids[2], ids[-1], [ids[2]], "spoken", allow_cross_scene=True), retry_of="c01")
        fake = FakeLLM([orig], [retry])
        _, c = self.run_pipe(fake, ep)
        self.assertEqual(c["c01"]["rejection_reasons"], ["validation:scene"])
        self.assertEqual(c["c01r"]["rejection_reasons"], ["validation:scene"])
        self.assertEqual(fake.calls.count("retry"), 1)

    def test_integrity_failures_are_not_retried(self):
        midline = P("e01_u025", "e01_u048", ["e01_u025"], "visual")            # dialogue_cut
        fake = FakeLLM([midline], [dict(P("e01_u031", "e01_u048", ["e01_u032"], "spoken"), retry_of="c01")])
        _, c = self.run_pipe(fake)
        self.assertEqual(c["c01"]["rejection_reasons"], ["validation:dialogue_cut"])
        self.assertNotIn("retry", fake.calls)
        self.assertNotIn("c01r", c)

    def test_retry_cannot_bypass_dialogue_cut(self):
        retry = dict(P("e01_u025", "e01_u048", ["e01_u025"], "visual"), retry_of="c01")
        _, c = self.run_pipe(FakeLLM([self.ALTAR_ORIG], [retry]))
        self.assertEqual(c["c01r"]["rejection_reasons"], ["validation:dialogue_cut"])

    def test_only_one_retry_per_candidate_and_no_retry_of_retry(self):
        bad_retry = dict(P("e01_u001", "e01_u029", ["e01_u008"], "spoken"), retry_of="c01")  # same mistake
        second = dict(self.ALTAR_RETRY)
        fake = FakeLLM([self.ALTAR_ORIG], [bad_retry, second])
        _, c = self.run_pipe(fake)
        self.assertEqual(fake.calls.count("retry"), 1)
        self.assertEqual(c["c01r"]["rejection_reasons"], ["validation:hook"])    # first answer used, second ignored
        self.assertNotIn("c01rr", c)
        raw = json.loads((self.tmp / "proposals_retry_raw.json").read_text(encoding="utf-8"))
        self.assertEqual(len(raw["ignored"]), 1)

    def test_original_rejection_kept_in_audit_and_failed_retry_not_repaired(self):
        bad = dict(P("e01_u001", "e01_u029", ["e01_u008"], "spoken"), retry_of="c01")
        final, c = self.run_pipe(FakeLLM([self.ALTAR_ORIG], [bad]))
        self.assertEqual(c["c01"]["status"], "rejected")
        self.assertEqual(c["c01"]["retried_as"], "c01r")
        self.assertIn("hook", c["c01"]["rejection_details"][0])
        self.assertEqual(c["c01r"]["status"], "rejected")
        self.assertEqual((c["c01r"]["start_unit_id"], c["c01r"]["start"]), ("e01_u001", 0.0))  # as proposed
        self.assertEqual(final["final"], [])
        for f in ("validation.json", "validation_retry.json", "proposals_retry_raw.json"):
            self.assertTrue((self.tmp / f).exists(), f)
        self.assertIn("hook 11.3", (self.tmp / "prompts" / "retry.txt").read_text(encoding="utf-8").replace("hook e01_u008 begins ", "hook "))

    def test_give_up_is_recorded(self):
        _, c = self.run_pipe(FakeLLM([self.ALTAR_ORIG], [{"retry_of": "c01", "give_up": True, "reason": "too slow"}]))
        self.assertEqual(c["c01"]["retry_declined"], "too slow")
        self.assertNotIn("c01r", c)

    def test_judge_hook_evidence_must_be_in_opening(self):
        late = lambda a, b, p: good_judge(a, b, hook={"score": 4, "evidence_unit_ids": ["e01_u048"]})
        _, c = self.run_pipe(FakeLLM([P("e01_u031", "e01_u048", ["e01_u032"], "spoken")], judge=late))
        self.assertEqual(c["c01"]["status"], "rejected")
        self.assertTrue(c["c01"]["rejection_reasons"][0].startswith("unsupported:judge_hook_evidence_starts"))


# ------------------------------------------------------------------ judge reliability (llm client)
def http_error(code, retry_after=None):
    hdrs = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("http://x", code, "err", hdrs, io.BytesIO(b""))


class JudgeReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cache, self._sleep, self._log = llm.CACHE_DIR, llm._sleep, llm.LOG_PATH
        llm.CACHE_DIR, llm.LOG_PATH = self.tmp / "cache", self.tmp / "llm.jsonl"
        self.slept = []
        llm._sleep = self.slept.append
        llm._last_call.clear()

    def tearDown(self):
        llm.CACHE_DIR, llm._sleep, llm.LOG_PATH = self._cache, self._sleep, self._log
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_gemini_judge_429_never_calls_groq(self):
        seen = []

        def transport(sc, messages):
            seen.append(sc["_provider"])
            raise http_error(429)
        with self.assertRaises(RuntimeError):
            llm.complete_json("judge", "p-429", cfg=CFG_GJ, transport=transport)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"gemini"})
        self.assertIsNone(llm.fallback_name(CFG_GJ, "judge"))
        self.assertEqual(llm.fallback_name(CFG, "propose"), "groq")  # proposer fallback stays on (and is logged)

    def test_retry_success_recorded_under_same_judge_model(self):
        n = {"i": 0}

        def transport(sc, messages):
            n["i"] += 1
            if n["i"] == 1:
                raise http_error(429, retry_after=7)
            return '{"ok": 1}', {"prompt_tokens": 3, "completion_tokens": 1}
        meta = {}
        self.assertEqual(llm.complete_json("judge", "p-retry", cfg=CFG_GJ, transport=transport, meta=meta), {"ok": 1})
        self.assertEqual((meta["provider"], meta["model"], meta["attempts"]), ("gemini", CFG_GJ["stages"]["judge"]["model"], 2))
        self.assertNotIn("fallback_from", meta)
        self.assertIn(7.0, self.slept)  # Retry-After honoured

    def test_retry_exhaustion_is_bounded_and_fails(self):
        def transport(sc, messages):
            raise http_error(503)
        meta = {}
        with self.assertRaises(RuntimeError):
            llm.complete_json("judge", "p-exhaust", cfg=CFG_GJ, transport=transport, meta=meta)
        self.assertTrue(meta.get("failed"))
        backoffs = [s for s in self.slept if s > 0]
        self.assertLessEqual(sum(backoffs), CFG["defaults"]["max_total_wait_s"] + 60)
        self.assertLessEqual(len(backoffs), CFG["defaults"]["max_retries"] * 2)


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class JudgeFailedAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cache, self._sleep, self._log = llm.CACHE_DIR, llm._sleep, llm.LOG_PATH
        llm.CACHE_DIR, llm.LOG_PATH = self.tmp / "cache", self.tmp / "llm.jsonl"
        llm._sleep = lambda s: None
        llm._last_call.clear()
        self.ep = load_episode(EP1)

    def tearDown(self):
        llm.CACHE_DIR, llm._sleep, llm.LOG_PATH = self._cache, self._sleep, self._log
        shutil.rmtree(self.tmp, ignore_errors=True)

    def transport_factory(self, fail_judge_for):
        props = {"candidates": [P("e01_u031", "e01_u048", ["e01_u032"], "spoken"),
                                P("e01_u069", "e01_u095", ["e01_u069"], "spoken")]}
        self.calls = []

        def transport(sc, messages):
            prompt = messages[-1]["content"]
            if "Find the best promo moments" in prompt:
                self.calls.append("propose")
                return json.dumps(props), {}
            import re
            ids = re.findall(r"^(e01_u\d{3}) \|", prompt, re.M)
            self.calls.append(f"judge:{ids[0]}")
            if fail_judge_for and ids[0] == fail_judge_for:
                raise http_error(429)
            return json.dumps(good_judge(ids[0], ids[-1])), {}
        return transport

    def llm_with(self, transport):
        def call(stage, prompt, system, use_cache, meta):
            return llm.complete_json(stage, prompt, system=system, cfg=CFG, transport=transport,
                                     use_cache=use_cache, meta=meta)
        return call

    def test_exhaustion_creates_judge_failed_and_resume_reruns_only_that_call(self):
        out = self.tmp / "batch"
        rows = run_batch([EP1], out, CFG, llm=self.llm_with(self.transport_factory("e01_u069")))
        self.assertEqual(rows[0]["status"], "partial_judge_failed")
        final = json.loads((out / "episode_001" / "final.json").read_text(encoding="utf-8"))
        c = {x["candidate_id"]: x for x in final["all_candidates"]}
        self.assertEqual(c["c02"]["rejection_reasons"], ["judge:failed"])          # no invented score
        self.assertIsNone(c["c02"]["final_score"])
        self.assertEqual(final["counts"]["judge_failed"], 1)
        # resume: provider healthy again; only the failed judge call is executed, the rest comes from cache
        rows = run_batch([EP1], out, CFG, llm=self.llm_with(self.transport_factory(None)))
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(self.calls, ["judge:e01_u069"])

    def test_resume_uses_stage_checkpoints_even_with_empty_cache(self):
        out = self.tmp / "batch"
        run_batch([EP1], out, CFG, llm=self.llm_with(self.transport_factory("e01_u069")))
        shutil.rmtree(llm.CACHE_DIR, ignore_errors=True)            # e.g. cache key scheme changed
        rows = run_batch([EP1], out, CFG, llm=self.llm_with(self.transport_factory(None)))
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(self.calls, ["judge:e01_u069"])            # proposals + good judgement from checkpoints
        final = json.loads((out / "episode_001" / "final.json").read_text(encoding="utf-8"))
        self.assertEqual(final["usage"]["checkpoint_reused"], 2)

    def test_changed_prompt_invalidates_checkpoint(self):
        out = self.tmp / "ep"
        pipeline.run_episode(self.ep, out, CFG, llm=self.llm_with(self.transport_factory(None)), resume=True)
        shutil.rmtree(llm.CACHE_DIR, ignore_errors=True)
        self.calls.clear()
        cfg2 = {**CFG, "propose": {**CFG.get("propose", {}), "max_candidates": 5}}  # different proposer prompt
        pipeline.run_episode(self.ep, out, cfg2, llm=self.llm_with(self.transport_factory(None)), resume=True)
        self.assertEqual(self.calls[0], "propose")                  # not reused: prompt hash changed
        self.assertNotIn("judge:e01_u031", self.calls[1:])          # judge prompts unchanged: reused


if __name__ == "__main__":
    unittest.main()


class QuotaAndCacheProvenanceTests(unittest.TestCase):
    """pilot_002: Gemini free tier = 20 judge requests/day; a Groq answer leaked through the cache."""
    DAILY = (b'[{"error": {"code": 429, "message": "You exceeded your current quota. * Quota exceeded for metric: '
             b'generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: x, '
             b'description: Generate content requests per day limit exceeded on Free Tier. Please retry in 12h16m31s."}}]')

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cache, self._sleep, self._log = llm.CACHE_DIR, llm._sleep, llm.LOG_PATH
        llm.CACHE_DIR, llm.LOG_PATH = self.tmp / "cache", self.tmp / "llm.jsonl"
        self.slept = []
        llm._sleep = self.slept.append
        llm._last_call.clear()
        llm._exhausted.clear()

    def tearDown(self):
        llm.CACHE_DIR, llm._sleep, llm.LOG_PATH = self._cache, self._sleep, self._log
        llm._exhausted.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_daily_quota_fails_fast_and_trips_circuit_breaker(self):
        calls = []

        def transport(sc, messages):
            calls.append(sc["_provider"])
            raise urllib.error.HTTPError("http://x", 429, "Too Many", {}, io.BytesIO(self.DAILY))
        meta = {}
        with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
            llm.complete_json("judge", "q1", cfg=CFG_GJ, transport=transport, meta=meta)
        self.assertEqual(calls, ["gemini"])                      # no retries
        self.assertEqual([s for s in self.slept if s > 0], [])   # no backoff sleeping
        self.assertTrue(meta["quota_exhausted"])
        with self.assertRaisesRegex(RuntimeError, "not retried"):
            llm.complete_json("judge", "q2", cfg=CFG_GJ, transport=transport)
        self.assertEqual(calls, ["gemini"])                      # breaker: second call never hits the provider

    def test_per_minute_429_still_backs_off(self):
        n = {"i": 0}

        def transport(sc, messages):
            n["i"] += 1
            if n["i"] < 3:
                raise urllib.error.HTTPError("http://x", 429, "Too Many", {}, io.BytesIO(b'{"error":"rate"}'))
            return '{"ok": 1}', {}
        self.assertEqual(llm.complete_json("judge", "q3", cfg=CFG_GJ, transport=transport), {"ok": 1})
        self.assertEqual(n["i"], 3)

    def test_fallback_answer_is_not_served_as_primary_from_cache(self):
        cfg = {"defaults": {"provider": "a", "fallback": "b", "max_retries": 1, "temperature": 0.0},
               "providers": {"a": {"model": "ma", "base_url": "ua", "api_key_env": "K"},
                             "b": {"model": "mb", "base_url": "ub", "api_key_env": "K"}}}
        state = {"a_down": True}
        seen = []

        def transport(sc, messages):
            seen.append(sc["_provider"])
            if sc["_provider"] == "a" and state["a_down"]:
                raise RuntimeError("down")
            return json.dumps({"by": sc["_provider"]}), {}
        self.assertEqual(llm.complete_json("judge", "same prompt", cfg=cfg, transport=transport), {"by": "b"})
        state["a_down"] = False
        cfg["defaults"]["fallback"] = ""                          # fallback now off: must ask the primary
        meta = {}
        self.assertEqual(llm.complete_json("judge", "same prompt", cfg=cfg, transport=transport, meta=meta), {"by": "a"})
        self.assertEqual(meta["provider"], "a")
        self.assertEqual(seen, ["a", "b", "a"])


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class ModelAwareCheckpointTests(unittest.TestCase):
    """Completion run: switching the judge model must re-judge everything, but never re-run the proposer."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ep = load_episode(EP1)
        self.props = [P("e01_u031", "e01_u048", ["e01_u032"], "spoken"), P("e01_u069", "e01_u095", ["e01_u069"], "spoken")]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake(self, judge_model):
        calls = []

        def call(stage, prompt, system, use_cache, meta):
            if stage == "propose":
                meta.update(provider="gemini", model=CFG["providers"]["gemini"]["model"])
                calls.append("propose")
                return {"candidates": self.props}
            meta.update(provider=judge_model[0], model=judge_model[1])
            import re
            ids = re.findall(r"^(e01_u\d{3}) \|", prompt, re.M)
            calls.append(f"judge:{ids[0]}")
            return good_judge(ids[0], ids[-1])
        return call, calls

    def test_judge_model_change_rejudges_all_and_reuses_proposals(self):
        cfg_old = copy.deepcopy(CFG_GJ)
        f1, c1 = self.fake(("gemini", "gemini-3.5-flash"))
        pipeline.run_episode(self.ep, self.tmp, cfg_old, llm=f1, resume=True)
        self.assertEqual(c1, ["propose", "judge:e01_u031", "judge:e01_u069"])
        want = (CFG["stages"]["judge"]["provider"], CFG["stages"]["judge"]["model"])
        f2, c2 = self.fake(want)
        final = pipeline.run_episode(self.ep, self.tmp, CFG, llm=f2, resume=True)
        self.assertEqual(c2, ["judge:e01_u031", "judge:e01_u069"])        # proposer reused, both re-judged
        judged = json.loads((self.tmp / "judged.json").read_text(encoding="utf-8"))
        self.assertEqual({(j["llm"]["provider"], j["llm"]["model"]) for j in judged}, {want})
        self.assertEqual(final["counts"]["final"], 2)
        f3, c3 = self.fake(want)                                            # same model again: nothing re-run
        pipeline.run_episode(self.ep, self.tmp, CFG, llm=f3, resume=True)
        self.assertEqual(c3, [])

    def test_legacy_checkpoint_without_hash_matches_saved_prompt_file(self):
        f1, c1 = self.fake(("groq", "openai/gpt-oss-120b"))
        pipeline.run_episode(self.ep, self.tmp, CFG, llm=f1, resume=True)
        raw = json.loads((self.tmp / "proposals_raw.json").read_text(encoding="utf-8"))
        raw.pop("prompt_hash")                                              # as written by pilot_002
        (self.tmp / "proposals_raw.json").write_text(json.dumps(raw), encoding="utf-8")
        f2, c2 = self.fake(("groq", "openai/gpt-oss-120b"))
        pipeline.run_episode(self.ep, self.tmp, CFG, llm=f2, resume=True)
        self.assertNotIn("propose", c2)

    def test_frozen_config_is_groq_judge_without_fallback(self):
        from promolense.llm import fallback_name, stage_cfg
        sc = stage_cfg(CFG, "judge")
        self.assertEqual((sc["_provider"], sc["model"]), ("groq", "openai/gpt-oss-120b"))
        self.assertIsNone(fallback_name(CFG, "judge"))
        self.assertEqual(stage_cfg(CFG, "propose")["_provider"], "gemini")


@unittest.skipUnless(EP1.exists(), "Ep 1 not present")
class RejudgeTests(unittest.TestCase):
    def test_rejudge_scores_old_windows_with_current_judge_and_never_touches_source(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ep = load_episode(EP1)
            props = [P("e01_u031", "e01_u048", ["e01_u032"], "spoken"), P("e01_u001", "e01_u029", ["e01_u008"], "spoken")]
            old = ModelAwareCheckpointTests.fake.__get__(type("X", (), {"props": props})())
            f1, _ = old(("gemini", "gemini-3.5-flash"))
            pipeline.run_episode(ep, tmp / "src", CFG_GJ, llm=f1)
            before = (tmp / "src" / "judged.json").read_text(encoding="utf-8")
            want = (CFG["stages"]["judge"]["provider"], CFG["stages"]["judge"]["model"])
            f2, c2 = old(want)
            final = pipeline.rejudge(ep, tmp / "src", tmp / "dst", CFG, llm=f2)
            self.assertEqual(c2, ["judge:e01_u031"])                     # only the valid window is judged
            self.assertEqual((tmp / "src" / "judged.json").read_text(encoding="utf-8"), before)
            judges = [m for m in final["usage"]["models"] if m.startswith("judge:")]
            self.assertEqual(judges, [f"judge:{want[0]}/{want[1]}"])                # all scoring by the current judge
            self.assertIn("propose:gemini/gemini-3.1-flash-lite", final["usage"]["models"])  # original provenance kept
            self.assertEqual({c["candidate_id"]: c["status"] for c in final["all_candidates"]},
                             {"c01": "final", "c02": "rejected"})
            with self.assertRaises(SystemExit):
                pipeline.rejudge(ep, tmp / "src", tmp / "dst", CFG, llm=f2)   # never overwrites
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RollingQuotaTests(unittest.TestCase):
    """batch_001: Groq tokens-per-day is a rolling window ('try again in 7m12s'): wait, don't trip the breaker."""
    TPD = (b'{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): '
           b'Limit 200000, Used 199000, Requested 2300. Please try again in 7m12.5s.","type":"tokens","code":"rate_limit_exceeded"}}')

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cache, self._sleep, self._log = llm.CACHE_DIR, llm._sleep, llm.LOG_PATH
        llm.CACHE_DIR, llm.LOG_PATH = self.tmp / "cache", self.tmp / "llm.jsonl"
        self.slept = []
        llm._sleep = self.slept.append
        llm._last_call.clear()
        llm._exhausted.clear()

    def tearDown(self):
        llm.CACHE_DIR, llm._sleep, llm.LOG_PATH = self._cache, self._sleep, self._log
        llm._exhausted.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hint_parser(self):
        self.assertEqual(llm._retry_hint_s("Please try again in 7m12.5s."), 432.5)
        self.assertEqual(llm._retry_hint_s("Please retry in 12h16m31.3s"), 44191.3)
        self.assertEqual(llm._retry_hint_s("try again in 45.2s"), 45.2)
        self.assertIsNone(llm._retry_hint_s("no hint"))

    def test_rolling_daily_window_is_waited_out(self):
        n = {"i": 0}

        def transport(sc, messages):
            n["i"] += 1
            if n["i"] == 1:
                raise urllib.error.HTTPError("http://x", 429, "Too Many", {}, io.BytesIO(self.TPD))
            return '{"ok": 1}', {}
        meta = {}
        self.assertEqual(llm.complete_json("judge", "tpd", cfg=CFG, transport=transport, meta=meta), {"ok": 1})
        self.assertIn(434.5, self.slept)          # waited the provider's hint (+2 s)
        self.assertEqual(llm._exhausted, {})      # breaker not tripped
        self.assertEqual(meta["attempts"], 2)

    def test_long_daily_hint_still_trips_breaker(self):
        body = self.TPD.replace(b"7m12.5s", b"5h2m1s")

        def transport(sc, messages):
            raise urllib.error.HTTPError("http://x", 429, "Too Many", {}, io.BytesIO(body))
        with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
            llm.complete_json("judge", "tpd2", cfg=CFG, transport=transport)
        self.assertIn("groq", llm._exhausted)
