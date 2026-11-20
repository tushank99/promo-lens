"""PromoLens command line. Run `python -m promolense --help` for all commands.

Common:
  python -m promolense batch --out outputs/batch_001          # every episode in episodes/
  python -m promolense run <episode.txt> --out outputs/ep01   # one episode
  python -m promolense qa outputs/batch_001                   # QA checklist for a batch
"""
import argparse
import sys

from .parser import parse_file


def main() -> int:
    ap = argparse.ArgumentParser(prog="promolense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="parse one script into timestamped units (no LLM)")
    p.add_argument("file")
    p.add_argument("--out")
    sub.add_parser("hello", help="check the LLM connection (one tiny call)")
    r = sub.add_parser("run", help="propose -> validate -> judge -> rank one episode")
    r.add_argument("file")
    r.add_argument("--out", required=True)
    r.add_argument("--no-cache", action="store_true", help="bypass the LLM cache (for stability runs)")
    r.add_argument("--format", default="auto", help="auto | screenplay_txt | subtitles | canonical_json")
    b = sub.add_parser("batch", help="run many episodes: isolated failures, resume, summary.csv")
    b.add_argument("--inputs", nargs="+", default=["episodes"])
    b.add_argument("--episodes", nargs="*", type=int, help="header episode numbers (default: all canonical)")
    b.add_argument("--out", required=True)
    b.add_argument("--no-cache", action="store_true")
    b.add_argument("--force", action="store_true", help="rerun episodes that are already complete")
    b.add_argument("--continuation", help="name an explicit config continuation (e.g. after a quota switch)")
    b.add_argument("--seed-from", help="copy matching episode folders (checkpoints) from another batch dir first")
    rj = sub.add_parser("rejudge", help="re-score an existing run's validated windows with the current judge")
    rj.add_argument("src")
    rj.add_argument("dst")
    rj.add_argument("--episode", default="episodes/episode-001-Episode 1671703.txt")
    qa = sub.add_parser("qa", help="final QA checklist for a batch dir")
    qa.add_argument("batch")
    rs = sub.add_parser("review-summary", help="compare human ratings with the judge's scores")
    rs.add_argument("--sheet", default="eval/ep01_review.csv")
    rs.add_argument("--runs-root", default="outputs/ep01")
    dv = sub.add_parser("discovery", help="moment discovery reliability across batch runs")
    dv.add_argument("--moments", default="eval/pilot_moments.csv")
    dv.add_argument("--runs", nargs="+", required=True, help="batch output dirs (repeat runs of the same episodes)")
    cb = sub.add_parser("compare-batches", help="two batch runs of the same episodes side by side")
    cb.add_argument("a")
    cb.add_argument("b")
    m = sub.add_parser("manifest", help="source files -> header episode, content hash, duplicate status")
    m.add_argument("paths", nargs="+", help="files or directories")
    m.add_argument("--out", default="outputs/manifest")
    n = sub.add_parser("normalize", help="any supported input -> canonical JSON (preview what the core will see)")
    n.add_argument("file")
    n.add_argument("--out", required=True)
    n.add_argument("--format", default="auto")
    e = sub.add_parser("eval", help="metrics for one run against hand labels")
    e.add_argument("run")
    e.add_argument("--episode", default="episodes/episode-001-Episode 1671703.txt")
    e.add_argument("--labels", default="eval/ep01_human.json")
    e.add_argument("--compare", help="another run dir of the same episode, for stability")
    v = sub.add_parser("validate-labels", help="run the validator on hand-labelled windows")
    v.add_argument("--episode", default="episodes/episode-001-Episode 1671703.txt")
    v.add_argument("--labels", default="eval/ep01_human.json")
    a = ap.parse_args()

    if a.cmd == "parse":
        ep = parse_file(a.file)
        kinds: dict[str, int] = {}
        for u in ep.units:
            kinds[u.type] = kinds.get(u.type, 0) + 1
        print(f"Episode {ep.number}: {len(ep.units)} units {kinds}, {len(ep.scenes)} scene(s)")
        flagged = [u for u in ep.units if u.flags and u.flags != ["missing_tone"]]
        print(f"flagged units: {len(flagged)}; exchanges: {len({u.exchange for u in ep.units if u.exchange})}")
        if ep.rejected:
            print("REJECTED:", ep.reject_reason)
        if a.out:
            ep.dump(a.out)
            print("wrote", a.out)
        return 1 if ep.rejected else 0

    if a.cmd == "run":
        from .llm import load_config
        from .adapters import load_episode
        from .pipeline import run_episode
        ep = load_episode(a.file, a.format)
        if ep.rejected:
            print("REJECTED:", ep.reject_reason)
            return 1
        cfg = load_config()
        final = run_episode(ep, a.out, cfg, use_cache=not a.no_cache)
        print("counts:", final["counts"])
        for k in final["final"]:
            print(f"  #{k['rank']} {k['candidate_id']} {k['start']}-{k['end']} ({k['duration']}s) "
                  f"score {k['final_score']}  hook={k['hook_unit_id']}  flags={k['flags']}")
        print("wrote", a.out)
        return 0

    if a.cmd == "batch":
        from pathlib import Path
        from .batch import run_batch
        from .llm import load_config
        files = []
        for x in a.inputs:
            px = Path(x)
            files += sorted(f for f in px.iterdir() if f.is_file()) if px.is_dir() else [px]
        if a.seed_from:
            import shutil
            Path(a.out).mkdir(parents=True, exist_ok=True)
            for d in sorted(Path(a.seed_from).glob("episode_*")):
                if (d / "final.json").exists() and not (Path(a.out) / d.name).exists():
                    shutil.copytree(d, Path(a.out) / d.name)
                    print("seeded", d.name, "from", a.seed_from)
        rows = run_batch(files, Path(a.out), load_config(), a.episodes, use_cache=not a.no_cache, force=a.force,
                         continuation=a.continuation)
        from collections import Counter
        print(f"{dict(Counter(r['status'] for r in rows))}; summary: {Path(a.out) / 'summary.csv'}")
        return 0 if all(r["status"] != "failed" for r in rows) else 1

    if a.cmd == "compare-batches":
        import json
        from pathlib import Path
        from .evaluate import compare_batches
        print(json.dumps(compare_batches(Path(a.a), Path(a.b)), indent=1, ensure_ascii=False))
        return 0

    if a.cmd in ("review-summary", "discovery"):
        import json
        from pathlib import Path
        from .evaluate import discovery, review_summary
        res = (review_summary(Path(a.sheet), Path(a.runs_root)) if a.cmd == "review-summary"
               else discovery(Path(a.moments), [Path(x) for x in a.runs]))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if a.cmd == "rejudge":
        from pathlib import Path
        from .adapters import load_episode
        from .llm import load_config
        from .pipeline import rejudge
        final = rejudge(load_episode(a.episode), Path(a.src), Path(a.dst), load_config())
        print("counts:", final["counts"], "| judge:", final["usage"]["models"])
        return 0

    if a.cmd == "qa":
        import json
        from pathlib import Path
        from .qa import run_qa
        rep_ = run_qa(Path(a.batch))
        print(json.dumps(rep_, indent=1, ensure_ascii=False))
        return 0 if rep_["passed"] else 1

    if a.cmd == "manifest":
        from pathlib import Path
        from .adapters import build_manifest, write_manifest
        files = []
        for x in a.paths:
            px = Path(x)
            files += sorted(f for f in px.iterdir() if f.is_file()) if px.is_dir() else [px]
        rows = build_manifest(files)
        write_manifest(rows, Path(a.out))
        counts = {}
        for r in rows:
            k = r["status"].split(":")[0]
            counts[k] = counts.get(k, 0) + 1
        print(f"{len(rows)} files: {counts}")
        for r in rows:
            if r["status"] != "canonical":
                print(f"  {r['source_file']}: {r['status']} {r.get('reject_reason') or r.get('error') or ''}")
        print("wrote", Path(a.out) / "manifest.csv")
        return 0

    if a.cmd == "normalize":
        import json
        from pathlib import Path
        from .adapters import load_episode
        from .schema import to_canonical
        ep = load_episode(a.file, a.format)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(to_canonical(ep), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{ep.episode_id}: {len(ep.units)} units, {len(ep.scenes)} scene(s), coverage {ep.visual_coverage}, "
              f"{'REJECTED: ' + ep.reject_reason if ep.rejected else 'ok'}; issues: {ep.issues}")
        print("wrote", a.out)
        return 1 if ep.rejected else 0

    if a.cmd == "eval":
        import json
        from pathlib import Path
        from .evaluate import evaluate_run, to_markdown
        ep = parse_file(a.episode)
        res = evaluate_run(ep, a.run, a.labels, a.compare)
        Path(a.run, "eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        md = to_markdown(res)
        Path(a.run, "eval.md").write_text(md, encoding="utf-8")
        print(md)
        return 0

    if a.cmd == "validate-labels":
        import json
        from .llm import load_config
        from .validator import validate
        ep = parse_file(a.episode)
        rules = load_config()["rules"]
        for l in json.load(open(a.labels, encoding="utf-8"))["labels"]:
            r = validate(ep, {k: l.get(k) for k in ("start_unit", "end_unit", "hook_unit", "cliffhanger_unit")}, rules)
            print(f"{l['id']} {l['name']:<32} {r.start}-{r.end} ({r.duration}s)  {'PASS' if r.ok else 'REJECT ' + ','.join(r.reasons)}")
            for d in r.details:
                print("     -", d)
        return 0

    if a.cmd == "hello":
        from .llm import complete_json
        print(complete_json("propose", 'Return {"ok": true, "word": "hello"}', use_cache=False))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
