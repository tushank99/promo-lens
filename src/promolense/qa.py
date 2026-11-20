"""Final QA checklist for a batch directory. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FINAL = ("episode", "counts", "final", "all_candidates", "usage")
REQUIRED_CAND = ("candidate_id", "start_unit_id", "end_unit_id", "status", "rejection_reasons", "segments")
DONE = ("complete", "complete_no_clips")


def run_qa(batch: Path) -> dict:
    checks, problems = {}, []
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((batch / "run_config.json").read_text(encoding="utf-8"))
    conts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(batch.glob("run_config.continuation-*.json"))]
    canon = sorted(m["header_episode"] for m in manifest if m["status"] == "canonical")
    dups = [m["source_file"] for m in manifest if m["status"].startswith("duplicate_of")]
    by_ep = {r["episode"]: r for r in summary}

    checks["canonical_episodes"] = len(canon)
    checks["all_canonical_accounted_for"] = sorted(by_ep) == canon
    if not checks["all_canonical_accounted_for"]:
        problems.append(f"summary episodes missing: {sorted(set(canon) - set(by_ep))}")
    checks["duplicates_skipped"] = dups
    checks["status_counts"] = {}
    for r in summary:
        checks["status_counts"][r["status"]] = checks["status_counts"].get(r["status"], 0) + 1
    not_done = {r["episode"]: r["status"] for r in summary if r["status"] not in DONE}
    checks["not_done"] = not_done

    judge_models = {(frozen["models"]["judge"]["provider"], frozen["models"]["judge"]["model"])}
    judge_models |= {(c["models"]["judge"]["provider"], c["models"]["judge"]["model"]) for c in conts}
    wrong_judge, bad_refs, schema, retries, finals = [], [], [], 0, 0
    for n in canon:
        d = batch / f"episode_{n:03d}"
        if not (d / "final.json").exists():
            continue
        f = json.loads((d / "final.json").read_text(encoding="utf-8"))
        miss = [k for k in REQUIRED_FINAL if k not in f]
        if miss:
            schema.append(f"ep {n}: final.json missing {miss}")
        norm = json.loads((d / "normalized.json").read_text(encoding="utf-8"))
        ids = {u["unit_id"] for sc in norm["scenes"] for u in sc["units"]}
        for c in f.get("all_candidates", []):
            m2 = [k for k in REQUIRED_CAND if k not in c]
            if m2:
                schema.append(f"ep {n} {c.get('candidate_id')}: missing {m2}")
            if c["status"] in ("final", "alternate"):
                refs = [c["start_unit_id"], c["end_unit_id"], c.get("hook_unit_id")] + list(
                    ((c.get("evidence_unit_ids") or {}).get("hook") or []))
                bad = [x for x in refs if x and x not in ids]
                if bad:
                    bad_refs.append(f"ep {n} {c['candidate_id']}: {bad}")
            retries += bool(c.get("retried_as"))
        finals += len(f.get("final", []))
        if (d / "judged.json").exists():
            for j in json.loads((d / "judged.json").read_text(encoding="utf-8")):
                pm = (j.get("llm", {}).get("provider"), j.get("llm", {}).get("model"))
                if j.get("raw") is not None and pm not in judge_models:
                    wrong_judge.append(f"ep {n} {j['candidate_id']}: {pm}")
    checks.update(schema_errors=schema, invalid_unit_refs=bad_refs, judge_calls_not_declared_model=wrong_judge,
                  declared_judge_models=sorted(map(list, judge_models)), retries_recorded=retries,
                  final_clips=finals, config_continuations=[c["config_id"] for c in conts])
    problems += schema + bad_refs + wrong_judge
    passed = checks["all_canonical_accounted_for"] and not problems
    rep = {"batch": str(batch), "passed": passed, "complete": passed and not not_done, "checks": checks,
           "problems": problems}
    (batch / "qa_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep
