"""Run several episodes with per-episode error isolation and a master summary.

Pilot-grade: sequential (no concurrency yet: that is Phase 6C). One failed episode never stops the batch;
rerunning resumes: episodes already complete are skipped (unless --force), so only failures are redone.
"""
from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

from .adapters import build_manifest, load_episode, write_manifest
from .pipeline import run_episode
from .schema import to_canonical


DONE = ("complete", "complete_no_clips")   # ran fine; no_clips = nothing passed (needs a look, not a rerun)


def select(manifest: list[dict], episodes: list[int] | None) -> list[dict]:
    canon = [r for r in manifest if r["status"] == "canonical"]
    if episodes is None:
        return canon
    by_ep = {r["header_episode"]: r for r in canon}
    missing = [e for e in episodes if e not in by_ep]
    if missing:
        raise SystemExit(f"episodes not found among canonical inputs: {missing}")
    return [by_ep[e] for e in episodes]


def run_batch(input_paths: list[Path], out_dir: Path, cfg: dict, episodes: list[int] | None = None,
              use_cache: bool = True, force: bool = False, llm=None, continuation: str | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(input_paths)
    write_manifest(manifest, out_dir)
    config_id = freeze_config(cfg, manifest, out_dir, continuation)
    rows = []
    prev = {}
    if (out_dir / "summary.json").exists():
        prev = {r["episode"]: r for r in json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))}
    for m in select(manifest, episodes):
        n = m["header_episode"]
        ep_dir = out_dir / f"episode_{n:03d}"
        # skip only work done under a config recorded for this batch (base or a named continuation); older rows
        # are re-run with resume, so unchanged stages come from checkpoints and only stale ones (e.g. judge) re-run
        done = ((ep_dir / "final.json").exists() and prev.get(n, {}).get("status") in DONE
                and bool(prev.get(n, {}).get("config_id")))
        if done and not force:
            rows.append({**prev[n], "note": "skipped: already complete"})
            print(f"Ep {n:>2}: skipped (already complete)")
            continue
        t0 = time.time()
        row = {"episode": n, "source_file": m["source_file"], "config_id": config_id, "status": "failed", "proposed": 0, "final": 0,
               "final_needing_review": 0, "rejected": 0, "best_score": None, "runtime_s": None, "llm_calls": 0,
               "tokens_in": 0, "tokens_out": 0, "retries": 0, "fallback_calls": 0, "error": ""}
        try:
            ep = load_episode(m["path"])
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "normalized.json").write_text(json.dumps(to_canonical(ep), indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
            if ep.rejected:
                raise ValueError(f"input rejected: {ep.reject_reason}")
            # resume from stage checkpoints unless this is a forced or cache-bypassing (fresh) run
            final = run_episode(ep, ep_dir, cfg, llm=llm, use_cache=use_cache, resume=use_cache and not force)
            c, u = final["counts"], final.get("usage", {})
            status = "complete" if c["final"] else "complete_no_clips"
            if u.get("judge_failed"):  # resumable: rerun redoes only the failed judge calls (the rest is cached)
                status = "partial_judge_failed"
            row.update(status=status, proposed=c["proposed"], final=c["final"],
                       final_needing_review=c.get("final_needing_review", 0),
                       rejected=c["rejected_at_validation"] + c["rejected_at_judge"] + c["rejected_at_floors"],
                       best_score=max((k["final_score"] for k in final["final"]), default=None),
                       llm_calls=u.get("calls", 0), tokens_in=u.get("tokens_in", 0), tokens_out=u.get("tokens_out", 0),
                       retries=u.get("retries", 0), fallback_calls=u.get("fallback_calls", 0))
        except Exception as e:  # isolate: record and continue
            row["error"] = f"{type(e).__name__}: {e}"[:500]
            if "quota exhausted" in str(e):   # not a permanent failure: resume later, possibly as a continuation
                row["status"] = "pending_quota"
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        row["runtime_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"Ep {n:>2}: {row['status']:<8} final={row['final']} (review {row['final_needing_review']}) "
              f"proposed={row['proposed']} best={row['best_score']} {row['runtime_s']}s {row['error']}")
        _write_summary(_merged(prev, rows), out_dir)  # checkpoint after every episode
    _write_summary(_merged(prev, rows), out_dir)
    return rows


def _merged(prev: dict, rows: list[dict]) -> list[dict]:
    """Summary keeps every episode ever run in this folder; rows from this run replace older ones."""
    by_ep = dict(prev)
    for r in rows:
        by_ep[r["episode"]] = r
    return [by_ep[k] for k in sorted(by_ep)]


FROZEN_KEYS = ("models", "prompt_template_sha256_16", "rules", "floors", "weights", "dedupe", "retry")


class ConfigChanged(SystemExit):
    pass


def config_record(cfg: dict, manifest: list[dict]) -> dict:
    """What produced a batch: models, prompt template hashes, rules, input hashes."""
    import hashlib
    from . import prompts
    from .llm import stage_cfg
    tpl = {k: hashlib.sha256(getattr(prompts, k).encode()).hexdigest()[:16]
           for k in ("PROPOSE_SYSTEM", "PROPOSE_TEMPLATE", "RETRY_TEMPLATE", "JUDGE_SYSTEM", "JUDGE_TEMPLATE")}
    models = {}
    if "defaults" in cfg:
        from .llm import fallback_name
        for st in ("propose", "judge"):
            sc = stage_cfg(cfg, st)
            models[st] = {"provider": sc.get("_provider"), "model": sc.get("model"),
                          "fallback": fallback_name(cfg, st), "min_interval_s": sc.get("min_interval_s"),
                          "temperature": sc.get("temperature")}
    rec = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": models, "prompt_template_sha256_16": tpl,
           "rules": cfg.get("rules"), "floors": cfg.get("floors"), "weights": cfg.get("weights"),
           "dedupe": cfg.get("dedupe"), "retry": cfg.get("retry"),
           "inputs": [{k: m.get(k) for k in ("source_file", "header_episode", "status", "content_sha256")}
                      for m in manifest]}
    return rec


def freeze_config(cfg: dict, manifest: list[dict], out_dir: Path, continuation: str | None = None) -> str:
    """run_config.json is written once and never overwritten. A run whose config differs is refused, unless it
    is an explicitly named continuation, which is recorded beside it. Returns the config id used."""
    rec = config_record(cfg, manifest)
    frozen_p = out_dir / "run_config.json"
    if not frozen_p.exists():
        frozen_p.write_text(json.dumps({**rec, "config_id": "base"}, indent=2, ensure_ascii=False), encoding="utf-8")
        return "base"
    frozen = json.loads(frozen_p.read_text(encoding="utf-8"))
    for c in sorted(out_dir.glob("run_config.continuation-*.json")):   # latest continuation is the active one
        frozen = json.loads(c.read_text(encoding="utf-8"))
    diff = {k: {"frozen": frozen.get(k), "now": rec.get(k)} for k in FROZEN_KEYS if frozen.get(k) != rec.get(k)}
    if not diff:
        return frozen.get("config_id", "base")
    if not continuation:
        raise ConfigChanged("config differs from the frozen run_config for this batch: " + ", ".join(diff) +
                            ". Revert it, or pass --continuation NAME to record an explicit continuation.")
    cid = f"continuation-{continuation}"
    (out_dir / f"run_config.{cid}.json").write_text(
        json.dumps({**rec, "config_id": cid, "changes_vs_previous": diff}, indent=2, ensure_ascii=False), encoding="utf-8")
    return cid


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ["episode", "config_id", "status", "proposed", "final", "final_needing_review", "rejected", "best_score", "runtime_s",
            "llm_calls", "tokens_in", "tokens_out", "retries", "fallback_calls", "error", "source_file"]
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
