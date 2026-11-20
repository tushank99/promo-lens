import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tempfile  # noqa: E402

from promolense import llm  # noqa: E402

llm.LOG_PATH = Path(tempfile.gettempdir()) / "promolense_test_llm.jsonl"  # keep the real call log clean

CFG = {"defaults": {"base_url": "x", "model": "test-model-xyz", "api_key_env": "NOPE",
                    "temperature": 0.0, "max_retries": 3}}


class LlmTests(unittest.TestCase):
    def test_cache_and_repair(self):
        calls = []

        def transport(sc, messages):
            calls.append(len(messages))
            if len(calls) == 1:
                return "not json at all", {}
            return '```json\n{"ok": true}\n```', {"prompt_tokens": 5, "completion_tokens": 2}

        prompt = "unit-test-prompt-unique-8431"
        r = llm.complete_json("propose", prompt, cfg=CFG, transport=transport)
        self.assertEqual(r, {"ok": True})
        self.assertEqual(len(calls), 2)  # one repair retry
        r2 = llm.complete_json("propose", prompt, cfg=CFG, transport=transport)
        self.assertEqual(r2, r)
        self.assertEqual(len(calls), 2)  # served from cache
        for f in (ROOT / "cache").glob("*.json"):
            if json.loads(f.read_text(encoding="utf-8")).get("result") == {"ok": True}:
                f.unlink()

    def test_stage_override(self):
        cfg = {"defaults": {"model": "a", "base_url": "u", "api_key_env": "K"}, "stages": {"judge": {"model": "b"}}}
        self.assertEqual(llm.stage_cfg(cfg, "propose")["model"], "a")
        self.assertEqual(llm.stage_cfg(cfg, "judge")["model"], "b")

    def test_fallback_to_secondary(self):
        cfg = {"defaults": {"provider": "a", "fallback": "b", "max_retries": 1},
               "providers": {"a": {"model": "ma", "base_url": "ua", "api_key_env": "K"},
                             "b": {"model": "mb", "base_url": "ub", "api_key_env": "K"}}}
        seen = []

        def transport(sc, messages):
            seen.append(sc["_provider"])
            if sc["_provider"] == "a":
                raise RuntimeError("quota")
            return '{"from": "b"}', {}

        r = llm.complete_json("propose", "fallback-test-prompt-5521", cfg=cfg, transport=transport, use_cache=False)
        self.assertEqual(r, {"from": "b"})
        self.assertEqual(seen, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
