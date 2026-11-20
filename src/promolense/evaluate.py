"""Phase 5 evaluation of one pipeline run against human labels. Code-only metrics.

Human-rated metrics (would publish, opening strength, promo angle, human ending type) come from a review
sheet the user fills in; judge calibration is computed only when that sheet has ratings.
IoU is a diagnostic, never the only criterion.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .pipeline import DIMENSIONS, iou, mmss

RECOVER_IOU = 0.5          # recovered if IoU >= this ...
RECOVER_COVER = 0.5        # ... or covers >= this share of the label AND
RECOVER_HOOK_S = 3.0       #     its hook starts within this many seconds of the label's hook
ACTIVITY = ("dialogue", "action", "sfx")


def _window(ep, s_id, e_id):
    ids = [u.id for u in ep.units]
    return ep.units[ids.index(s_id):ids.index(e_id) + 1]


def opening(ep, c) -> dict:
    """Hook delay and silent opening for a timed candidate."""
    units = {u.id: u for u in ep.units}
    w = _window(ep, c["start_unit_id"], c["end_unit_id"])
    first_act = min((u.start for u in w if u.type in ACTIVITY), default=None)
    silent = round(first_act - c["start"], 2) if first_act is not None else round(c["duration"], 2)
    first_kind = next((u.type for u in sorted(w, key=lambda u: u.start) if u.type in ACTIVITY), None)
    o = {
        "hook_delay_s": round(units[c["hook_unit_id"]].start - c["start"], 2) if c.get("hook_unit_id") in units else None,
        "judge_hook_delay_s": (round(units[c["judge_hook_unit_id"]].start - c["start"], 2)
                               if c.get("judge_hook_unit_id") in units else None),
        "silent_opening_s": silent,
        # code can't tell a compelling visual from dead screen time: 'camera_only' is for the human to rate
        "opening_type": first_kind if silent <= 1.0 else "camera_only",
    }
    return o


def match(ep, label, c) -> dict:
    units = {u.id: u for u in ep.units}
    ids = [u.id for u in ep.units]
    inter = max(0.0, min(label["end"], c["end"]) - max(label["start"], c["start"]))
    cover = inter / label["duration"] if label["duration"] else 0.0
    hc = units.get(c.get("judge_hook_unit_id") or c.get("hook_unit_id"))
    hl = units.get(label.get("hook_unit"))
    hook_gap = abs(hc.start - hl.start) if hc and hl else None
    m = {
        "candidate_id": c["candidate_id"], "status": c["status"],
        "iou": round(iou(label["start"], label["end"], c["start"], c["end"]), 3),
        "label_coverage": round(cover, 3),
        "hook_gap_s": round(hook_gap, 2) if hook_gap is not None else None,
        "start_delta_s": round(c["start"] - label["start"], 2), "end_delta_s": round(c["end"] - label["end"], 2),
        "start_delta_units": ids.index(c["start_unit_id"]) - ids.index(label["start_unit"]),
        "end_delta_units": ids.index(c["end_unit_id"]) - ids.index(label["end_unit"]),
    }
    m["recovered"] = m["iou"] >= RECOVER_IOU or (cover >= RECOVER_COVER and hook_gap is not None
                                                 and hook_gap <= RECOVER_HOOK_S)
    m["boundary_within_1_unit"] = abs(m["start_delta_units"]) <= 1 and abs(m["end_delta_units"]) <= 1
    return m


def coverage(cands, ep_end: float) -> dict:
    """Share of each third of the episode touched by these candidates' windows."""
    thirds = [(0, ep_end / 3), (ep_end / 3, 2 * ep_end / 3), (2 * ep_end / 3, ep_end)]
    out = {}
    for name, (a, b) in zip(("first_third", "middle_third", "last_third"), thirds):
        spans = sorted((max(a, c["start"]), min(b, c["end"])) for c in cands if c["start"] < b and c["end"] > a)
        cov, cur = 0.0, a
        for s, e in spans:
            s = max(s, cur)
            if e > s:
                cov += e - s
                cur = e
        out[name] = {"range": f"{mmss(a)}-{mmss(b)}", "share_covered": round(cov / (b - a), 2),
                     "candidates": [c["candidate_id"] for c in cands if c["start"] < b and c["end"] > a]}
    return out


def evaluate_run(ep, run_dir: str | Path, labels_path: str | Path, compare_dir: str | Path | None = None) -> dict:
    run = Path(run_dir)
    final = json.loads((run / "final.json").read_text(encoding="utf-8"))
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
    cands = final["all_candidates"]
    timed = [c for c in cands if c.get("start") is not None]
    finals = [c for c in cands if c["status"] == "final"]
    judged = [c for c in cands if c.get("scores")]
    ep_end = max(u.end for u in ep.units if u.end is not None)

    # --- per-label recovery (proposed / passed / final), boundaries, rejection loss
    per_label = []
    for l in labels:
        ms = sorted((match(ep, l, c) for c in timed), key=lambda m: (-m["recovered"], -m["iou"]))
        overlapping = [m for m in ms if m["iou"] > 0]
        best = ms[0] if ms else None
        rec_any = [m for m in ms if m["recovered"]]
        rec_final = [m for m in rec_any if m["status"] == "final"]
        rejected_overlap = [m for m in overlapping if m["status"] == "rejected"]
        row = {
            "label": l["id"], "name": l["name"], "window": f"{mmss(l['start'])}-{mmss(l['end'])} ({l['duration']}s)",
            "meets_20_60s": l.get("meets_20_60s"),
            "discovered": bool(overlapping), "proposal_recovered": bool(rec_any), "final_recovered": bool(rec_final),
            "recovered_by": [m["candidate_id"] for m in rec_any], "final_recovered_by": [m["candidate_id"] for m in rec_final],
            "best_match": best, "boundary_within_1_unit": bool(best and best["recovered"] and best["boundary_within_1_unit"]),
            "rejection_loss": [{"candidate_id": m["candidate_id"], "iou": m["iou"],
                                "reasons": next(c["rejection_reasons"] for c in cands if c["candidate_id"] == m["candidate_id"])}
                               for m in rejected_overlap],
        }
        if not l.get("meets_20_60s", True):
            row["policy_note"] = ("Label is below the 20 s minimum, so no valid candidate can match it exactly. "
                                  "Discovered = a proposal overlaps it; see rejection_loss for why it was dropped.")
        per_label.append(row)

    # --- openings, coverage, judge spread, ending types, angle proxies
    openings = {c["candidate_id"]: opening(ep, c) for c in timed}
    spread = {}
    for d in DIMENSIONS:
        v = [c["scores"][d] for c in judged]
        if v:
            spread[d] = {"min": min(v), "max": max(v), "mean": round(statistics.mean(v), 2),
                         "stdev": round(statistics.pstdev(v), 2)}
    ending_types = {}
    for c in judged:
        ending_types[c.get("ending_type")] = ending_types.get(c.get("ending_type"), 0) + 1
    pair_overlaps = [{"a": a["candidate_id"], "b": b["candidate_id"], "iou": round(iou(a["start"], a["end"], b["start"], b["end"]), 2)}
                     for i, a in enumerate(finals) for b in finals[i + 1:]
                     if iou(a["start"], a["end"], b["start"], b["end"]) > 0]

    res = {
        "run": str(run), "episode": final["episode"], "counts": final["counts"],
        "labels": per_label,
        "summary": {
            "labels_in_policy": sum(1 for l in labels if l.get("meets_20_60s", True)),
            "proposal_recovery": f"{sum(r['proposal_recovered'] for r in per_label if r['meets_20_60s'])}/"
                                 f"{sum(1 for r in per_label if r['meets_20_60s'])} in-policy labels",
            "final_recovery": f"{sum(r['final_recovered'] for r in per_label if r['meets_20_60s'])}/"
                              f"{sum(1 for r in per_label if r['meets_20_60s'])} in-policy labels",
            "boundary_within_1_unit": f"{sum(r['boundary_within_1_unit'] for r in per_label if r['meets_20_60s'])}/"
                                      f"{sum(1 for r in per_label if r['meets_20_60s'])}",
        },
        "openings": openings,
        "time_coverage": {"proposed_timed": coverage(timed, ep_end), "final": coverage(finals, ep_end)},
        "judge_score_spread": spread, "ending_types": ending_types,
        "final_angle_proxies": {
            "distinct_hook_types": sorted({c.get("hook_type") for c in finals}),
            "distinct_emotion_types": sorted({c.get("emotion_type") for c in finals}),
            "overlapping_final_pairs": pair_overlaps,
        },
    }
    if compare_dir:
        res["stability"] = stability(run, Path(compare_dir), ep)
    return res


def _same_moment(ep, a, b) -> bool:
    units = {u.id: u for u in ep.units}
    if iou(a["start"], a["end"], b["start"], b["end"]) >= RECOVER_IOU:
        return True
    ha = units.get(a.get("judge_hook_unit_id") or a.get("hook_unit_id"))
    hb = units.get(b.get("judge_hook_unit_id") or b.get("hook_unit_id"))
    return bool(ha and hb and abs(ha.start - hb.start) <= RECOVER_HOOK_S
                and iou(a["start"], a["end"], b["start"], b["end"]) > 0)


def stability(run_a: Path, run_b: Path, ep) -> dict:
    fa = json.loads((run_a / "final.json").read_text(encoding="utf-8"))
    fb = json.loads((run_b / "final.json").read_text(encoding="utf-8"))
    A = [c for c in fa["all_candidates"] if c["status"] == "final"]
    B = [c for c in fb["all_candidates"] if c["status"] == "final"]
    PA = [c for c in fa["all_candidates"] if c.get("start") is not None]
    PB = [c for c in fb["all_candidates"] if c.get("start") is not None]
    pairs = [(a["candidate_id"], b["candidate_id"]) for a in A for b in B if _same_moment(ep, a, b)]
    return {
        "compare_to": str(run_b),
        "final_a": [f"{c['candidate_id']} {mmss(c['start'])}-{mmss(c['end'])}" for c in A],
        "final_b": [f"{c['candidate_id']} {mmss(c['start'])}-{mmss(c['end'])}" for c in B],
        "final_a_found_in_b": f"{len({p[0] for p in pairs})}/{len(A)}",
        "final_b_found_in_a": f"{len({p[1] for p in pairs})}/{len(B)}",
        "matched_pairs": pairs,
        "proposals_a_found_in_b": f"{sum(any(_same_moment(ep, a, b) for b in PB) for a in PA)}/{len(PA)}",
        "top1_same_moment": bool(A and B and _same_moment(ep, min(A, key=lambda c: c['rank']), min(B, key=lambda c: c['rank']))),
    }


def write_review_sheet(ep, runs: list[Path], path: Path) -> int:
    """One row per timed candidate across runs, for the human to rate BLIND (no judge scores shown, to avoid
    anchoring; they are joined back by run + candidate_id). Existing ratings are preserved."""
    units = {u.id: u for u in ep.units}
    old = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                old[(r["run"], r["candidate_id"])] = r
    human = ["would_publish(y/n)", "opening_strength(1-5)", "promo_angle", "ending_type_human", "notes", "rater"]
    # ending_type_human: cliffhanger | reveal | confrontation | emotional_resolution | transition | abrupt
    rows = []
    for run in runs:
        final = json.loads((run / "final.json").read_text(encoding="utf-8"))
        for c in final["all_candidates"]:
            if c.get("start") is None:
                continue
            h = units.get(c.get("judge_hook_unit_id") or c["hook_unit_id"])
            ev = ((c.get("evidence_unit_ids") or {}).get("ending") or [None])[-1]
            e = units.get(ev or c["end_unit_id"])
            r = {"run": run.name, "candidate_id": c["candidate_id"], "status": c["status"],
                 "rejected_for": ",".join(c.get("rejection_reasons", [])),
                 "window": f"{mmss(c['start'])}-{mmss(c['end'])}", "duration": c["duration"],
                 "units": f"{c['start_unit_id']}..{c['end_unit_id']}",
                 "hook_text": h.text if h else "", "ending_text": e.text if e else ""}
            prev = old.get((run.name, c["candidate_id"]), {})
            for k in human:
                r[k] = prev.get(k, "")
            rows.append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def to_markdown(res: dict) -> str:
    L = [f"# Phase 5 eval: {res['run']}", "", f"Counts: {res['counts']}", "",
         "## Label recovery", "",
         "| Label | Window | Discovered | Proposal recovered | Final recovered | Best match (IoU / coverage / hook gap) | Boundary +/-1 unit | Rejection loss |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["labels"]:
        b = r["best_match"]
        bm = f"{b['candidate_id']} ({b['status']}): {b['iou']} / {b['label_coverage']} / {b['hook_gap_s']}s" if b else "none"
        loss = "; ".join(f"{x['candidate_id']} {','.join(x['reasons'])}" for x in r["rejection_loss"]) or "none"
        L.append(f"| {r['label']} {r['name']} | {r['window']} | {r['discovered']} | {r['proposal_recovered']} "
                 f"| {r['final_recovered']} | {bm} | {r['boundary_within_1_unit']} | {loss} |")
    for r in res["labels"]:
        if r.get("policy_note"):
            L += ["", f"**{r['label']}:** {r['policy_note']}"]
    L += ["", f"Summary: {res['summary']}", "", "## Openings (hook delay, silent opening)", "",
          "| Candidate | Hook delay (proposer) | Hook delay (judge) | Silent opening | Opening type |", "|---|---|---|---|---|"]
    for cid, o in res["openings"].items():
        f = lambda x: "n/a" if x is None else f"{x}s"
        L.append(f"| {cid} | {f(o['hook_delay_s'])} | {f(o['judge_hook_delay_s'])} | {f(o['silent_opening_s'])} | {o['opening_type']} |")
    L += ["", "## Time coverage (share of each third touched)", ""]
    for k, cov in res["time_coverage"].items():
        L.append(f"- {k}: " + ", ".join(f"{n} {v['range']}: {v['share_covered']} {v['candidates']}" for n, v in cov.items()))
    L += ["", "## Judge", "", f"- Score spread: {res['judge_score_spread']}", f"- Ending types: {res['ending_types']}",
          "", "## Final-list diversity proxies (human promo_angle in the review sheet is the real test)", "",
          f"- {res['final_angle_proxies']}"]
    if res.get("stability"):
        s = res["stability"]
        L += ["", "## Stability (cache bypassed)", "", f"- This run final: {s['final_a']}", f"- Rerun final: {s['final_b']}",
              f"- Final found in rerun: {s['final_a_found_in_b']}; rerun final found here: {s['final_b_found_in_a']}",
              f"- Proposals found in rerun: {s['proposals_a_found_in_b']}; same #1 moment: {s['top1_same_moment']}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ Phase 5.1: human review summary
def review_summary(sheet: Path, runs_root: Path) -> dict:
    """Join the blind human ratings to the judge's output and report agreement. Unrated rows are ignored."""
    with open(sheet, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    rated = [r for r in rows if (r.get("would_publish(y/n)") or "").strip().lower() in ("y", "n")]
    finals = {}
    for r in rated:
        fpath = runs_root / r["run"] / "final.json"
        if r["run"] not in finals and fpath.exists():
            finals[r["run"]] = {c["candidate_id"]: c for c in json.loads(fpath.read_text(encoding="utf-8"))["all_candidates"]}
    out = {"rows": len(rows), "rated": len(rated)}
    if not rated:
        out["note"] = "no ratings yet: fill would_publish(y/n), opening_strength(1-5), promo_angle, ending_type_human"
        return out
    pub = [r["would_publish(y/n)"].strip().lower() == "y" for r in rated]
    out["publish_rate"] = round(sum(pub) / len(pub), 2)
    opens = [int(r["opening_strength(1-5)"]) for r in rated if (r.get("opening_strength(1-5)") or "").strip().isdigit()]
    out["opening_strength_mean"] = round(statistics.mean(opens), 2) if opens else None
    # ending-type agreement (judge vs human), only where both exist
    # human vocabulary -> judge vocabulary (judge: cliffhanger | payoff | emotional_resolution | transition)
    to_judge = {"reveal": "payoff", "confrontation": "payoff", "payoff": "payoff", "abrupt": "transition",
                "cliffhanger": "cliffhanger", "emotional_resolution": "emotional_resolution", "transition": "transition"}
    agree, pairs = [], []
    for r in rated:
        c = finals.get(r["run"], {}).get(r["candidate_id"], {})
        h = (r.get("ending_type_human") or "").strip().lower()
        if h and c.get("ending_type"):
            agree.append(to_judge.get(h, h) == c["ending_type"])
            pairs.append(f"{r['run']}/{r['candidate_id']}: human {h} vs judge {c['ending_type']}")
    out["ending_type_agreement"] = f"{sum(agree)}/{len(agree)}" if agree else None
    out["ending_type_pairs"] = pairs
    out["raters"] = sorted({(r.get("rater") or "unspecified") for r in rated})
    # angle diversity per run (final clips only)
    angles = {}
    for r in rated:
        if r["status"] == "final" and (r.get("promo_angle") or "").strip():
            angles.setdefault(r["run"], []).append(r["promo_angle"].strip().lower())
    out["angles_per_run"] = {k: {"clips": len(v), "distinct": len(set(v)), "angles": sorted(set(v))} for k, v in angles.items()}
    # judge calibration: do clips humans would publish score higher? (pairwise win rate, 0.5 = chance)
    scored = [(finals.get(r["run"], {}).get(r["candidate_id"], {}).get("final_score"), p) for r, p in zip(rated, pub)]
    scored = [(s, p) for s, p in scored if s is not None]
    ys, ns = [s for s, p in scored if p], [s for s, p in scored if not p]
    if ys and ns:
        wins = sum((y > n) + 0.5 * (y == n) for y in ys for n in ns)
        out["judge_pairwise_win_rate"] = round(wins / (len(ys) * len(ns)), 2)
        out["judge_mean_score_publish_vs_not"] = [round(statistics.mean(ys), 1), round(statistics.mean(ns), 1)]
    else:
        out["judge_pairwise_win_rate"] = None
        out["judge_note"] = "needs at least one 'y' and one 'n' among judged clips"
    rej = [r for r in rated if r["status"] == "rejected"]
    out["rejected_but_publishable"] = [f"{r['run']}/{r['candidate_id']} ({r.get('rejected_for')})"
                                       for r in rej if r["would_publish(y/n)"].strip().lower() == "y"]
    return out


# ------------------------------------------------------------------ discovery reliability
def discovery(moments_csv: Path, run_dirs: list[Path], min_cover: float = 0.6) -> dict:
    """For each human-marked compelling moment: in how many runs did at least one VALID (pass/review)
    proposal cover >= min_cover of the moment? Repeatability != reliability: this measures the latter."""
    with open(moments_csv, encoding="utf-8", newline="") as f:
        moments = [r for r in csv.DictReader(f) if (r.get("start_unit") or "").strip()]
    out = []
    for m in moments:
        ep = int(m["episode"])
        hits, detail = 0, []
        for rd in run_dirs:
            fpath = rd / f"episode_{ep:03d}" / "final.json"
            if not fpath.exists():
                detail.append(f"{rd.name}: no output")
                continue
            final = json.loads(fpath.read_text(encoding="utf-8"))
            ms, me = _moment_times(rd / f"episode_{ep:03d}", m["start_unit"], m["end_unit"])
            if ms is None:
                detail.append(f"{rd.name}: moment units not found in normalized.json")
                continue
            best_valid, best_any = None, None
            for c in final["all_candidates"]:
                if c.get("start") is None:
                    continue
                inter = max(0.0, min(me, c["end"]) - max(ms, c["start"]))
                cover = inter / (me - ms) if me > ms else 0.0
                rec = (cover, c["candidate_id"], c["status"], ",".join(c.get("rejection_reasons", [])))
                if best_any is None or cover > best_any[0]:
                    best_any = rec
                if c.get("rejected_at") != "validation" and (best_valid is None or cover > best_valid[0]):
                    best_valid = rec
            hit = bool(best_valid and best_valid[0] >= min_cover)
            hits += hit
            txt = f"{rd.name}: valid best " + (f"{best_valid[1]} cover {best_valid[0]:.2f}" if best_valid else "none")
            if best_any and (not best_valid or best_any[0] > best_valid[0]):
                txt += f"; better cover by rejected {best_any[1]} ({best_any[0]:.2f}, {best_any[3]})"
            detail.append(txt)
        out.append({"episode": ep, "moment": m.get("moment_name"), "units": f"{m['start_unit']}..{m['end_unit']}",
                    "discovered_runs": f"{hits}/{len(run_dirs)}", "detail": detail})
    return {"min_cover": min_cover, "moments": out}


def _moment_times(ep_dir: Path, s_id: str, e_id: str):
    """Times of a moment from the episode's normalized units (written by the batch)."""
    p = ep_dir / "normalized.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text(encoding="utf-8"))
    units = [u for sc in d["scenes"] for u in sc["units"]]
    ids = [u["unit_id"] for u in units]
    if s_id not in ids or e_id not in ids:
        return None, None
    w = units[ids.index(s_id):ids.index(e_id) + 1]
    return min(u["start"] for u in w), max(u["end"] for u in w)


# ------------------------------------------------------------------ batch-vs-batch comparison
def compare_batches(a: Path, b: Path) -> dict:
    """Side-by-side of two batch runs of the same episodes: outcomes, rejection reasons, hooks, retries, cost."""
    def load(run: Path):
        out = {}
        for d in sorted(run.glob("episode_*")):
            f = d / "final.json"
            if f.exists():
                out[int(d.name.split("_")[1])] = json.loads(f.read_text(encoding="utf-8"))
        return out

    def reasons(finals):
        cnt = {}
        for fin in finals.values():
            for c in fin["all_candidates"]:
                for r in c.get("rejection_reasons", []):
                    k = r.split("<")[0] if r.startswith("floor:") else r.split("_starts_")[0]
                    cnt[k] = cnt.get(k, 0) + 1
        return dict(sorted(cnt.items(), key=lambda x: -x[1]))

    def modal(finals, final_only):
        cnt = {}
        for fin in finals.values():
            for c in fin["all_candidates"]:
                if final_only and c["status"] != "final":
                    continue
                m = (c.get("hook") or {}).get("modality") or "legacy/none"
                cnt[m] = cnt.get(m, 0) + 1
        return cnt

    A, B = load(a), load(b)
    eps = sorted(set(A) | set(B))
    per_ep = []
    for e in eps:
        fa, fb = A.get(e, {}), B.get(e, {})
        row = {"episode": e}
        for tag, fin in (("a", fa), ("b", fb)):
            c = fin.get("counts", {})
            u = fin.get("usage", {})
            row[tag] = {"proposed": c.get("proposed"), "final": c.get("final"), "review": c.get("final_needing_review"),
                        "retried": c.get("retried"), "recovered": c.get("retry_recovered"),
                        "best": max((k["final_score"] for k in fin.get("final", [])), default=None),
                        "finals": [f"{k['candidate_id']} {mmss(k['start'])}-{mmss(k['end'])} "
                                   f"{(k.get('hook') or {}).get('modality') or ''}".strip() for k in fin.get("final", [])],
                        "wall_s": u.get("wall_s"), "retries": u.get("retries"), "waited_s": u.get("waited_s"),
                        "fallback_calls": u.get("fallback_calls"), "models": u.get("models")}
        per_ep.append(row)
    tot = lambda F, k: sum((f.get("counts", {}).get(k) or 0) for f in F.values())
    return {"a": str(a), "b": str(b), "episodes": per_ep,
            "totals": {t: {k: tot(F, k) for k in ("proposed", "final", "final_needing_review", "retried",
                                                   "retry_recovered", "judge_failed")}
                       for t, F in (("a", A), ("b", B))},
            "rejection_reasons": {"a": reasons(A), "b": reasons(B)},
            "hook_modality_all": {"a": modal(A, False), "b": modal(B, False)},
            "hook_modality_final": {"a": modal(A, True), "b": modal(B, True)}}
