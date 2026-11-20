"""Phase 4 pipeline for one episode:
propose -> validate -> judge -> floors/score -> dedupe -> final.

Every stage writes its own file under out_dir and nothing is overwritten by a later stage:
  proposals_raw.json  validation.json  judged.json  final.json  final.md  prompts/*.txt
Rejected candidates keep their reasons. Nothing is repaired.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .parser import Episode
from .prompts import JUDGE_SYSTEM, JUDGE_TEMPLATE, PROPOSE_SYSTEM, PROPOSE_TEMPLATE, RETRY_TEMPLATE
from .validator import _union_gaps, validate

DIMENSIONS = ("hook", "clarity", "emotion", "escalation", "ending")
EPS = 1e-6


# ---------------------------------------------------------------- helpers
def mmss(t: float | None) -> str:
    if t is None:
        return "?"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def render_unit(u, t0: float | None = None) -> str:
    when = f"{u.start:.1f}-{u.end:.1f}" if t0 is None else f"{u.start - t0:.1f}-{u.end - t0:.1f}"
    if u.type == "dialogue":
        tone = f" ({u.tone})" if u.tone else ""
        body = f'{u.speaker}{tone}: "{u.text}"'
    else:
        body = u.text
    return f"{u.id} | {when} | {u.type.upper()} | {body}"


def timeline_facts(ep: Episode, rules: dict) -> str:
    """Facts the validator enforces, stated up front so the proposer stops guessing."""
    timed = [u for u in ep.units if u.start is not None]
    end = max(u.end for u in timed)
    L = [f"- Episode runs {mmss(0)}-{mmss(end)} ({end:.1f} s)."]
    for sc in ep.scenes:
        su = [u for u in timed if u.scene == sc.id]
        if su:
            L.append(f"- Scene {sc.id} ({sc.heading}): {min(u.start for u in su):.1f}-{max(u.end for u in su):.1f} s, "
                     f"units {su[0].id}..{su[-1].id}.")
    gaps, visual = [], []
    for sc in ep.scenes:
        su = [u for u in timed if u.scene == sc.id]
        if not su:
            continue
        lo, hi = min(u.start for u in su), max(u.end for u in su)
        for g0, g1 in _union_gaps([(u.start, u.end) for u in su], lo, hi):
            if g1 - g0 > rules["dead_air_review_s"] + EPS:
                gaps.append(f"{g0:.1f}-{g1:.1f} s ({g1 - g0:.1f} s)")
        for g0, g1 in _union_gaps([(u.start, u.end) for u in su if u.type != "camera"], lo, hi):
            if g1 - g0 > rules["dead_air_review_s"] + EPS:
                visual.append(f"{g0:.1f}-{g1:.1f} s")
    if getattr(ep, "visual_coverage", "full") == "full":
        L.append(f"- Gaps with NOTHING scripted over {rules['dead_air_review_s']:g} s (allowed, flagged for human "
                 f"review): " + ("; ".join(gaps) if gaps else "none"))
        L.append("- Camera-only stretches (visual storytelling, no dialogue/action; fully allowed): "
                 + ("; ".join(visual) if visual else "none"))
    else:
        L.append(f"- Source is {ep.visual_coverage}: only dialogue is known; visual content between lines is unknown.")
    ex: dict[str, list[str]] = {}
    for u in timed:
        if u.exchange:
            ex.setdefault(u.exchange, []).append(u.id)
    L.append("- Overlapping dialogue exchanges (include all members or none): "
             + ("; ".join(" + ".join(v) for v in ex.values()) if ex else "none"))
    return "\n".join(L)


def render_episode(ep: Episode) -> str:
    return "\n".join(render_unit(u) for u in ep.units if u.start is not None)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def phash(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\n" + prompt).encode("utf-8")).hexdigest()[:16]


def _same_prompt(saved: dict, h: str, prompt_file: Path, full_text: str) -> bool:
    """Prompt identity: recorded hash, or (checkpoints written before hashes existed) the saved prompt file."""
    if saved.get("prompt_hash"):
        return saved["prompt_hash"] == h
    return prompt_file.exists() and prompt_file.read_text(encoding="utf-8") == full_text


def _same_model(saved_llm: dict, want: tuple | None) -> bool:
    return want is None or (saved_llm.get("provider"), saved_llm.get("model")) == tuple(want)


def _checkpoint(path: Path, h: str, prompt_file: Path, full_text: str, want: tuple | None):
    """Reuse a saved LLM response only if the same prompt AND the same provider/model produced it."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "response" not in d or d.get("error"):
        return None
    if not _same_prompt(d, h, prompt_file, full_text) or not _same_model(d.get("llm", {}), want):
        return None
    return d


def stage_model(cfg: dict, stage: str) -> tuple | None:
    if "defaults" not in cfg:
        return None
    from .llm import stage_cfg
    sc = stage_cfg(cfg, stage)
    return (sc.get("_provider"), sc.get("model"))


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > EPS else 0.0


# ---------------------------------------------------------------- stage 1: propose
def propose(ep: Episode, out: Path, llm, cfg: dict, use_cache: bool, resume: bool = False) -> list[dict]:
    max_c = cfg.get("propose", {}).get("max_candidates", 6)
    prompt = PROPOSE_TEMPLATE.format(logline=ep.logline or "(none)", continuity=ep.continuity or "(none)",
                                     timeline=timeline_facts(ep, cfg["rules"]),
                                     units=render_episode(ep), max_candidates=max_c)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    pf, full = out / "prompts" / "propose.txt", PROPOSE_SYSTEM + "\n\n" + prompt
    h = phash(PROPOSE_SYSTEM, prompt)
    ck = _checkpoint(out / "proposals_raw.json", h, pf, full, stage_model(cfg, "propose")) if resume else None
    pf.write_text(full, encoding="utf-8")
    if ck:
        meta, resp = {**ck.get("llm", {}), "checkpoint": True}, ck["response"]
    else:
        meta = {}
        resp = llm("propose", prompt, PROPOSE_SYSTEM, use_cache, meta)
    raw = resp.get("candidates") if isinstance(resp, dict) else None
    cands = raw if isinstance(raw, list) else []
    _write(out / "proposals_raw.json", {"episode": ep.number, "prompt_hash": h, "llm": meta, "response": resp,
                                        "note": None if isinstance(raw, list) else "response had no 'candidates' list"})
    cleaned = []
    for i, c in enumerate(cands[:max_c], 1):
        cleaned.append({"candidate_id": f"c{i:02d}", "raw": c})
    if len(cands) > max_c:
        for i, c in enumerate(cands[max_c:], max_c + 1):
            cleaned.append({"candidate_id": f"c{i:02d}", "raw": c, "over_cap": True})
    return cleaned


# ---------------------------------------------------------------- stage 2: validate
def normalize_proposal(c) -> tuple[dict | None, str | None]:
    """Structured hook {unit_ids, modality, description, evidence}; legacy hook_unit_id still accepted."""
    if not isinstance(c, dict) or not isinstance(c.get("start_unit_id"), str) or not isinstance(c.get("end_unit_id"), str):
        return None, "needs string start_unit_id and end_unit_id"
    h = c.get("hook")
    if isinstance(h, dict):
        ids = h.get("unit_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            return None, "hook.unit_ids must be a non-empty list of unit ids"
        if not isinstance(h.get("modality"), str):
            return None, "hook.modality must be visual, spoken or audio"
        hook = {"unit_ids": ids, "modality": h["modality"].strip().lower(),
                "description": h.get("description"), "evidence": h.get("evidence")}
    elif isinstance(c.get("hook_unit_id"), str):
        hook = {"unit_ids": [c["hook_unit_id"]], "modality": None, "description": None, "evidence": None}
    else:
        return None, "needs hook.unit_ids (or legacy hook_unit_id)"
    cl = c.get("cliffhanger_unit_id")
    return {"start_unit_id": c["start_unit_id"], "end_unit_id": c["end_unit_id"], "hook": hook,
            "cliffhanger_unit_id": cl if isinstance(cl, str) else None, "reason": c.get("reason")}, None


def validate_all(ep: Episode, proposals: list[dict], out: Path, rules: dict,
                 fname: str = "validation.json") -> list[dict]:
    rows = []
    for p in proposals:
        c = p["raw"]
        row = {"candidate_id": p["candidate_id"], "proposal": c}
        if p.get("retry_of"):
            row["retry_of"] = p["retry_of"]
        norm, err = normalize_proposal(c)
        if p.get("over_cap"):
            row.update(ok=False, reasons=["over_candidate_cap"], details=["more than max_candidates proposed"])
        elif err:
            row.update(ok=False, reasons=["malformed_proposal"], details=[err])
        else:
            row["norm"] = norm
            r = validate(ep, {"start_unit": norm["start_unit_id"], "end_unit": norm["end_unit_id"],
                              "hook_units": norm["hook"]["unit_ids"], "hook_modality": norm["hook"]["modality"],
                              "cliffhanger_unit": norm["cliffhanger_unit_id"],
                              # scene policy comes from config only; a proposal cannot switch it on
                              "allow_cross_scene": bool(rules.get("allow_cross_scene", False))}, rules)
            row.update(ok=r.ok, status=r.status, reasons=r.reasons, details=r.details, warnings=r.warnings,
                       review=r.review, review_details=r.review_details, hook_delay=r.hook_delay,
                       start=r.start, end=r.end, duration=r.duration, segments=r.segments)
        rows.append(row)
    _write(out / fname, rows)
    return rows


# ---------------------------------------------------------------- stage 2b: one bounded re-proposal
def retryable(v: dict, cfg: dict) -> bool:
    rc = cfg.get("retry", {})
    allowed = set(rc.get("retryable_reasons", ["hook", "hook_modality", "duration", "scene", "cliffhanger"]))
    return (rc.get("enabled", True) and not v["ok"] and "norm" in v and not v.get("retry_of")
            and bool(v["reasons"]) and set(v["reasons"]) <= allowed)


def _retry_context(ep: Episode, v: dict, rules: dict) -> str:
    ids = [u.id for u in ep.units]
    U = {u.id: u for u in ep.units}
    n = v["norm"]
    h0 = min((U[h] for h in n["hook"]["unit_ids"] if h in U), key=lambda u: u.start, default=None)
    anchor = h0 or U.get(n["start_unit_id"])
    scene = anchor.scene if anchor else None
    su = [u for u in ep.units if u.scene == scene and u.start is not None]
    if len(su) > 120:  # long scene: neighbourhood around the window
        a = max(0, ids.index(n["start_unit_id"]) - 40)
        b = min(len(ids), ids.index(n["end_unit_id"]) + 40)
        su = [u for u in ep.units[a:b] if u.scene == scene and u.start is not None]
    L = [f"- {v['candidate_id']}: window {n['start_unit_id']}..{n['end_unit_id']} "
         f"({v.get('start')}-{v.get('end')} s, {v.get('duration')} s); hook {n['hook']['unit_ids']} "
         f"modality {n['hook']['modality']} (delay {v.get('hook_delay')} s)",
         f"  FAILED: {', '.join(v['reasons'])}: " + " | ".join(v["details"]),
         f"  Original idea: {n.get('reason')}"]
    if su:
        L.append(f"  Scene {scene}: units {su[0].id}..{su[-1].id}, {min(u.start for u in su):.1f}-"
                 f"{max(u.end for u in su):.1f} s. Units of that scene:")
        L += ["    " + render_unit(u) for u in su]
    return "\n".join(L)


def repropose(ep: Episode, vrows: list[dict], out: Path, llm, cfg: dict, use_cache: bool, resume: bool = False):
    """One re-proposal round for window-selection failures only. Returns (proposals, declined{orig: reason})."""
    todo = [v for v in vrows if retryable(v, cfg)]
    if not todo:
        return [], {}
    prompt = RETRY_TEMPLATE.format(logline=ep.logline or "(none)",
                                   rejected="\n\n".join(_retry_context(ep, v, cfg["rules"]) for v in todo))
    pf, full = out / "prompts" / "retry.txt", PROPOSE_SYSTEM + "\n\n" + prompt
    allowed = {v["candidate_id"] for v in todo}
    h = phash(PROPOSE_SYSTEM, prompt)
    ck = _checkpoint(out / "proposals_retry_raw.json", h, pf, full, stage_model(cfg, "propose")) if resume else None
    pf.write_text(full, encoding="utf-8")
    if ck:
        meta, resp = {**ck.get("llm", {}), "checkpoint": True}, ck["response"]
    else:
        meta = {}
        try:
            resp = llm("propose", prompt, PROPOSE_SYSTEM, use_cache, meta)
        except Exception as e:  # logged; originals simply stay rejected
            _write(out / "proposals_retry_raw.json", {"prompt_hash": h, "llm": meta, "error": str(e),
                                                      "asked_for": sorted(allowed)})
            return [], {}
    raw = resp.get("retries") if isinstance(resp, dict) else None
    props, declined, ignored, seen = [], {}, [], set()
    for r in (raw if isinstance(raw, list) else []):
        orig = r.get("retry_of") if isinstance(r, dict) else None
        if orig not in allowed or orig in seen:  # unknown, or a second retry for the same candidate
            ignored.append(r)
            continue
        seen.add(orig)
        if r.get("give_up"):
            declined[orig] = r.get("reason") or "gave up"
            continue
        props.append({"candidate_id": f"{orig}r", "raw": r, "retry_of": orig})
    _write(out / "proposals_retry_raw.json", {"prompt_hash": h, "llm": meta, "asked_for": sorted(allowed), "response": resp,
                                              "declined": declined, "ignored": ignored})
    return props, declined


# ---------------------------------------------------------------- stage 3: judge
def _scores_ok(j) -> str | None:
    if not isinstance(j, dict):
        return "judge returned a non-object"
    for d in DIMENSIONS:
        blk = j.get(d)
        if not isinstance(blk, dict) or not isinstance(blk.get("score"), int) or isinstance(blk.get("score"), bool) \
                or not 0 <= blk["score"] <= 5:
            return f"'{d}.score' must be an integer 0-5"
        if not isinstance(blk.get("evidence_unit_ids", []), list):
            return f"'{d}.evidence_unit_ids' must be a list"
    return None


def judge_one(ep: Episode, vrow: dict, out: Path, llm, use_cache: bool, previous: dict | None = None,
              want: tuple | None = None) -> dict:
    cid = vrow["candidate_id"]
    ids = [u.id for u in ep.units]
    a, b = ids.index(vrow["segments"][0]["start_unit"]), ids.index(vrow["segments"][0]["end_unit"])
    window = ep.units[a:b + 1]
    cs, ce = vrow["start"], vrow["end"]
    prompt = JUDGE_TEMPLATE.format(
        logline=ep.logline or "(none)", continuity=ep.continuity or "(none)",
        clip_start=cs, clip_end=ce, duration=vrow["duration"],
        units="\n".join(render_unit(u, cs) for u in window))
    pf, full = out / "prompts" / f"judge_{cid}.txt", JUDGE_SYSTEM + "\n\n" + prompt
    h = phash(JUDGE_SYSTEM, prompt)
    reuse = (previous and not previous.get("error") and previous.get("raw") is not None
             and _same_prompt(previous, h, pf, full) and _same_model(previous.get("llm", {}), want))
    pf.write_text(full, encoding="utf-8")
    if reuse:
        return {**previous, "prompt_hash": h, "llm": {**previous.get("llm", {}), "checkpoint": True}}
    meta: dict = {}
    row = {"candidate_id": cid, "prompt_hash": h, "llm": meta}
    try:
        j = llm("judge", prompt, JUDGE_SYSTEM, use_cache, meta)
    except Exception as e:  # logged, never silently dropped
        row.update(raw=None, error=f"judge call failed: {e}", error_kind="judge_failed")
        return row
    row["raw"] = j
    err = _scores_ok(j)
    if err:
        row.update(error=f"invalid judge output: {err}", error_kind="judge_invalid_output")
    return row


def check_judgement(ep: Episode, vrow: dict, jrow: dict, rules: dict) -> dict:
    """Code-side verification of the judge. Returns parsed scores, verified evidence and flags."""
    j = jrow["raw"]
    ids = [u.id for u in ep.units]
    seg = vrow["segments"][0]
    window = {u.id: u for u in ep.units[ids.index(seg["start_unit"]):ids.index(seg["end_unit"]) + 1]}
    cs, dur = vrow["start"], vrow["duration"]
    final_from = cs + (1 - rules["cliffhanger_final_fraction"]) * dur
    flags, evidence, scores, invalid = [], {}, {}, {}
    for d in DIMENSIONS:
        scores[d] = j[d]["score"]
        cited = [x for x in j[d].get("evidence_unit_ids", []) if isinstance(x, str)]
        evidence[d] = [x for x in cited if x in window]
        bad = [x for x in cited if x not in window]
        if bad:
            invalid[d] = bad
    ending_type = j["ending"].get("type") if isinstance(j["ending"].get("type"), str) else "transition"
    if scores["ending"] > 1 and ending_type not in ("transition", "neither"):
        in_final = [x for x in evidence["ending"] if window[x].start >= final_from - EPS]
        if not in_final:
            flags.append("ending_evidence_not_in_final_20pct")
            scores["ending"] = 1  # penalised, not rejected: ending is a ranking factor
    hook_j = j.get("hook_unit_id") if j.get("hook_unit_id") in window else None
    hook_ev = [window[x].start for x in evidence["hook"]]
    hook_ev_delay = round(min(hook_ev) - cs, 3) if hook_ev else None
    return {
        "judge_hook_evidence_delay": hook_ev_delay,
        "scores": scores, "evidence": evidence, "invalid_evidence": invalid, "flags": flags,
        "ending_type": ending_type, "judge_hook_unit_id": hook_j,
        "hook_type": j.get("hook_type"), "emotion_type": j.get("emotion_type"), "trope": j.get("trope"),
        "central_conflict": j.get("central_conflict"), "spoils_main_twist": bool(j.get("spoils_main_twist")),
    }


# ---------------------------------------------------------------- stage 4: floors, score, dedupe
def score(scores: dict, weights: dict, spoils: bool) -> float:
    s = sum(weights[d] * scores[d] / 5 for d in DIMENSIONS) * 100
    if spoils:
        s *= weights.get("spoiler_penalty", 1.0)
    return round(s, 1)


def build_final(ep: Episode, vrows: list[dict], jrows: dict, cfg: dict, out: Path) -> dict:
    rules, floors, weights, dd = cfg["rules"], cfg["floors"], cfg["weights"], cfg["dedupe"]
    units = {u.id: u for u in ep.units}
    cands: list[dict] = []
    for v in vrows:
        c = v["proposal"] if isinstance(v["proposal"], dict) else {}
        n = v.get("norm") or {}
        hook = dict(n.get("hook") or {})
        hids = [h for h in hook.get("unit_ids", []) if h in units]
        first_hook = min(hids, key=lambda h: units[h].start) if hids else (c.get("hook_unit_id") or None)
        if hook:
            hook["delay_s"] = v.get("hook_delay")
        rec = {
            "candidate_id": v["candidate_id"], "episode": ep.number,
            "retry_of": v.get("retry_of"), "retried_as": v.get("retried_as"), "retry_declined": v.get("retry_declined"),
            "start_unit_id": c.get("start_unit_id"), "end_unit_id": c.get("end_unit_id"),
            "hook_unit_id": first_hook, "hook": hook or None, "cliffhanger_unit_id": n.get("cliffhanger_unit_id"),
            "segments": v.get("segments", []), "start": v.get("start"), "end": v.get("end"),
            "duration": v.get("duration"), "proposer_reason": c.get("reason"),
            "hook_type": None, "emotion_type": None, "scores": None, "evidence_unit_ids": None,
            "spoils_main_twist": None, "flags": list(v.get("warnings", [])), "rejection_reasons": [],
            "final_score": None, "status": "rejected",
            "review_status": v.get("status", "reject"),          # pass | review | reject (validator)
            "review_flags": list(v.get("review", [])), "review_details": list(v.get("review_details", [])),
        }
        if not v["ok"]:
            rec["rejected_at"] = "validation"
            rec["rejection_reasons"] = [f"validation:{r}" for r in v["reasons"]]
            rec["rejection_details"] = v["details"]
            cands.append(rec)
            continue
        jrow = jrows[v["candidate_id"]]
        if jrow.get("error") or jrow.get("raw") is None:
            rec["rejected_at"] = "judge"
            rec["rejection_reasons"] = [f"judge:{'failed' if jrow.get('error_kind') == 'judge_failed' else 'invalid_output'}"]
            rec["rejection_details"] = [jrow.get("error", "no judge output")]
            cands.append(rec)
            continue
        chk = check_judgement(ep, v, jrow, rules)
        rec.update(hook_type=chk["hook_type"], emotion_type=chk["emotion_type"], trope=chk["trope"],
                   scores=chk["scores"], evidence_unit_ids=chk["evidence"], spoils_main_twist=chk["spoils_main_twist"],
                   central_conflict=chk["central_conflict"], ending_type=chk["ending_type"],
                   judge_hook_unit_id=chk["judge_hook_unit_id"])
        rec["flags"] += chk["flags"]
        if chk["invalid_evidence"]:
            rec["flags"].append("judge_cited_ids_outside_clip")
            rec["invalid_evidence"] = chk["invalid_evidence"]
        # hook agreement is informational: independent judge vs proposer
        if chk["judge_hook_unit_id"] and rec["hook_unit_id"] in units:
            gap = abs(units[chk["judge_hook_unit_id"]].start - units[rec["hook_unit_id"]].start)
            if gap > dd["hook_same_within_s"]:
                rec["flags"].append("judge_and_proposer_disagree_on_hook")
        if chk["spoils_main_twist"]:
            rec["flags"].append("spoils_main_twist")
        if jrow.get("llm", {}).get("fallback_from"):  # scores came from a different judge model: not comparable
            rec["flags"].append(f"judged_by_fallback:{jrow['llm'].get('provider')}/{jrow['llm'].get('model')}")
        reasons = []
        for dim in ("hook", "clarity"):                      # hard gates, with verified evidence
            if chk["scores"][dim] < floors[dim]:
                reasons.append(f"floor:{dim}<{floors[dim]}")
            elif not chk["evidence"][dim]:
                reasons.append(f"unsupported:{dim}_has_no_valid_evidence")
        # the opening must be independently compelling: the judge's own hook evidence has to start in the window
        d_ev = chk["judge_hook_evidence_delay"]
        if d_ev is not None and d_ev > rules["hook_start_within_s"] + EPS and not any(r.startswith("floor:hook") for r in reasons):
            reasons.append(f"unsupported:judge_hook_evidence_starts_{d_ev}s_in")
        rec["judge_hook_evidence_delay_s"] = d_ev
        rec["final_score"] = score(chk["scores"], weights, chk["spoils_main_twist"])
        if reasons:
            rec["rejected_at"] = "floors"
            rec["rejection_reasons"] = reasons
        else:
            rec["status"] = "passed"
        cands.append(rec)

    passed = sorted([c for c in cands if c["status"] == "passed"], key=lambda c: -c["final_score"])
    kept: list[dict] = []
    for c in passed:
        dup_of = None
        for k in kept:
            ov = iou(c["start"], c["end"], k["start"], k["end"])
            if ov < dd["iou_threshold"]:
                continue
            ha = units[c["judge_hook_unit_id"] or c["hook_unit_id"]].start
            hb = units[k["judge_hook_unit_id"] or k["hook_unit_id"]].start
            if abs(ha - hb) <= dd["hook_same_within_s"] or ov >= dd["near_identical_iou"]:
                dup_of = k
                break
        if dup_of:
            c["status"] = "alternate"
            c["alternate_of"] = dup_of["candidate_id"]
            dup_of.setdefault("alternates", []).append(c["candidate_id"])
        else:
            overl = [k["candidate_id"] for k in kept
                     if iou(c["start"], c["end"], k["start"], k["end"]) >= dd["iou_threshold"]]
            if overl:
                c["flags"].append("overlaps_but_different_hook:" + ",".join(overl))
            c["status"] = "final"
            kept.append(c)
    for i, k in enumerate(kept, 1):
        k["rank"] = i
    final = {
        "episode": ep.number, "logline": ep.logline,
        "counts": {"proposed": len(cands),
                   "rejected_at_validation": sum(c.get("rejected_at") == "validation" for c in cands),
                   "rejected_at_judge": sum(c.get("rejected_at") == "judge" for c in cands),
                   "rejected_at_floors": sum(c.get("rejected_at") == "floors" for c in cands),
                   "alternates": sum(c["status"] == "alternate" for c in cands), "final": len(kept),
                   "retried": sum(bool(c.get("retried_as")) for c in cands),
                   "retry_declined": sum(bool(c.get("retry_declined")) for c in cands),
                   "retry_recovered": sum(bool(c.get("retry_of")) and c["status"] in ("final", "alternate") for c in cands),
                   "judge_failed": sum("judge:failed" in c["rejection_reasons"] for c in cands),
                   "final_needing_review": sum(k["review_status"] == "review" for k in kept)},
        "final": kept, "all_candidates": cands,
    }
    _write(out / "final.json", final)
    return final


def to_markdown(ep: Episode, final: dict) -> str:
    units = {u.id: u for u in ep.units}
    L = [f"# Episode {ep.number}: promo candidates", "", f"*{ep.logline}*", "",
         f"Proposed {final['counts']['proposed']}, final {final['counts']['final']}, "
         f"alternates {final['counts']['alternates']}.", ""]
    if not final["final"]:
        L.append("**No promo passed the checks for this episode.**")
    for k in final["final"]:
        h = units.get(k["judge_hook_unit_id"] or k["hook_unit_id"])
        ev_end = ((k["evidence_unit_ids"] or {}).get("ending") or [None])[-1]
        e = units.get(ev_end or k["end_unit_id"])
        L += [f"## #{k['rank']}{'  [REVIEW]' if k['review_status'] == 'review' else ''}  Ep {k['episode']}  {mmss(k['start'])} to {mmss(k['end'])}  ({k['duration']}s)  "
              f"score {k['final_score']}",
              f"- Segments: " + "; ".join(f"{s['start_unit']}..{s['end_unit']} ({mmss(s['start'])}-{mmss(s['end'])})"
                                           for s in k["segments"]),
              f"- Hook ({k['hook_type']}; proposer: {(k.get('hook') or {}).get('modality')}, "
              f"delay {(k.get('hook') or {}).get('delay_s')}s): {h.text if h else '?'}  [{h.id if h else ''} at {mmss(h.start) if h else ''}]"
              + (f"  (retry of {k['retry_of']})" if k.get("retry_of") else ""),
              f"- Ending ({k.get('ending_type')}): {e.text if e else '?'}  [{e.id if e else ''}]",
              f"- Central conflict: {k.get('central_conflict')}",
              f"- Why it works (proposer): {k.get('proposer_reason')}",
              f"- Scores: " + ", ".join(f"{d} {v}" for d, v in k["scores"].items()) +
              f"  | emotion: {k['emotion_type']} | trope: {k.get('trope')}",
              f"- Review: {'NEEDS REVIEW: ' + '; '.join(k['review_details']) if k['review_status'] == 'review' else 'pass'}",
              f"- Flags: {', '.join(k['flags']) or 'none'}",
              f"- Alternates: {', '.join(k.get('alternates', [])) or 'none'}", ""]
    rej = [c for c in final["all_candidates"] if c["status"] == "rejected"]
    if rej:
        L += ["## Rejected", ""]
        for c in rej:
            link = (f" -> retried as {c['retried_as']}" if c.get("retried_as") else
                    f" -> retry declined: {c['retry_declined']}" if c.get("retry_declined") else
                    f" (retry of {c['retry_of']})" if c.get("retry_of") else "")
            L.append(f"- {c['candidate_id']} ({c['start_unit_id']}..{c['end_unit_id']}, "
                     f"{c['duration']}s): {', '.join(c['rejection_reasons'])}{link}"
                     + (f" | {'; '.join(c.get('rejection_details', []))}" if c.get("rejection_details") else ""))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- orchestration
def default_llm(cfg: dict):
    from .llm import complete_json

    def call(stage, prompt, system, use_cache, meta):
        return complete_json(stage, prompt, system=system, cfg=cfg, use_cache=use_cache, meta=meta)
    return call


def run_episode(ep: Episode, out_dir: str | Path, cfg: dict, llm=None, use_cache: bool = True,
                resume: bool = False) -> dict:
    """resume=True reuses each stage's saved LLM output when its prompt hash is unchanged, so only failed or
    changed stages call the model again (independent of the response cache)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    llm = llm or default_llm(cfg)
    if ep.rejected:
        final = {"episode": ep.number, "rejected_episode": ep.reject_reason, "final": [], "all_candidates": [],
                 "counts": {"proposed": 0, "final": 0, "alternates": 0}}
        _write(out / "final.json", final)
        return final
    t0 = time.time()
    prev_judged = {}
    if resume and (out / "judged.json").exists():
        try:
            prev_judged = {j["candidate_id"]: j for j in json.loads((out / "judged.json").read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError):
            prev_judged = {}
    proposals = propose(ep, out, llm, cfg, use_cache, resume)
    vrows = validate_all(ep, proposals, out, cfg["rules"])
    retry_props, declined = repropose(ep, vrows, out, llm, cfg, use_cache, resume)
    rvrows = validate_all(ep, retry_props, out, cfg["rules"], fname="validation_retry.json") if retry_props else []
    for v in vrows:  # originals stay rejected; the link is kept in the audit trail
        if f"{v['candidate_id']}r" in {r["candidate_id"] for r in rvrows}:
            v["retried_as"] = f"{v['candidate_id']}r"
        if v["candidate_id"] in declined:
            v["retry_declined"] = declined[v["candidate_id"]]
    vrows = vrows + rvrows
    jrows = {}
    for v in vrows:
        if v["ok"]:
            jrows[v["candidate_id"]] = judge_one(ep, v, out, llm, use_cache, prev_judged.get(v["candidate_id"]),
                                                 stage_model(cfg, "judge"))
    _write(out / "judged.json", list(jrows.values()))
    final = build_final(ep, vrows, jrows, cfg, out)
    final["usage"] = usage_summary(out, jrows, time.time() - t0)
    _write(out / "final.json", final)
    (out / "final.md").write_text(to_markdown(ep, final), encoding="utf-8")
    return final


def usage_summary(out: Path, jrows: dict, wall_s: float) -> dict:
    metas = [("propose", json.loads((out / "proposals_raw.json").read_text(encoding="utf-8")).get("llm", {}))]
    if (out / "proposals_retry_raw.json").exists():
        metas.append(("retry", json.loads((out / "proposals_retry_raw.json").read_text(encoding="utf-8")).get("llm", {})))
    metas += [("judge", j.get("llm", {})) for j in jrows.values()]
    u = {"wall_s": round(wall_s, 1), "calls": len(metas), "cached_calls": sum(bool(m.get("cached")) for _, m in metas),
         "checkpoint_reused": sum(bool(m.get("checkpoint")) for _, m in metas),
         "tokens_in": sum(m.get("tokens_in", 0) for _, m in metas),
         "tokens_out": sum(m.get("tokens_out", 0) for _, m in metas),
         "retries": sum(max(0, m.get("attempts", 1) - 1) for _, m in metas),
         "fallback_calls": sum(bool(m.get("fallback_from")) for _, m in metas),
         "llm_secs": round(sum(m.get("secs", 0) for _, m in metas), 1),
         "models": sorted({f"{st}:{m.get('provider')}/{m.get('model')}" for st, m in metas if m.get("model")}),
         "judge_errors": sum(1 for j in jrows.values() if j.get("error")),
         "judge_failed": sum(1 for j in jrows.values() if j.get("error_kind") == "judge_failed"),
         "waited_s": round(sum(m.get("waited_s", 0) for _, m in metas), 1)}
    return u


def rejudge(ep: Episode, src: Path, dst: Path, cfg: dict, llm=None) -> dict:
    """Judge-only re-scoring: take the windows a past run validated and score them with the CURRENT judge into a new
    folder. Proposals and validation are copied as they were (audit); the source folder is never modified."""
    import shutil
    if dst.exists():
        raise SystemExit(f"{dst} exists; rejudge never overwrites")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("judged.json", "final.json", "final.md", "eval.*"))
    llm = llm or default_llm(cfg)
    t0 = time.time()
    vrows = json.loads((dst / "validation.json").read_text(encoding="utf-8"))
    if (dst / "validation_retry.json").exists():
        vrows += json.loads((dst / "validation_retry.json").read_text(encoding="utf-8"))
    jrows = {v["candidate_id"]: judge_one(ep, v, dst, llm, True, None, stage_model(cfg, "judge"))
             for v in vrows if v["ok"]}
    _write(dst / "judged.json", list(jrows.values()))
    final = build_final(ep, vrows, jrows, cfg, dst)
    final["rejudged_from"] = str(src)
    final["usage"] = usage_summary(dst, jrows, time.time() - t0)
    _write(dst / "final.json", final)
    (dst / "final.md").write_text(to_markdown(ep, final), encoding="utf-8")
    return final
