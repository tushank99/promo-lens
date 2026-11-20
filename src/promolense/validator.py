"""Hard validity rules for a promo candidate. Pure code. Reject, never repair.

A candidate is unit IDs, never times:
  {"start_unit", "end_unit", "hook_units": [...], "hook_modality": "visual"|"spoken"|"audio",
   "cliffhanger_unit", "allow_cross_scene": bool (policy from config, never from the proposer)}
Legacy single "hook_unit" is still accepted (modality is then not checked).
Hook delay = earliest hook unit start - clip start, computed here, never supplied by the LLM.
Clip time = min(start) .. max(end) over every unit in the window (file order, inclusive).
The validator never changes the candidate or the episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EPS = 1e-6

DEFAULT_RULES = {
    "min_clip_s": 20, "max_clip_s": 60, "hook_start_within_s": 3.0,
    "cliffhanger_final_fraction": 0.20,
    # A stretch covered by ANY unit (dialogue, action, camera, sfx) is occupied: camera units carry visual
    # storytelling in these scripts. An uncovered gap > dead_air_review_s is flagged for review, never rejected:
    # the validator checks structure, it does not judge cinematic meaning.
    "dead_air_review_s": 3.0,
}

# Reason codes
IDS, ORDER, TIMESTAMP, DURATION = "ids", "order", "timestamp", "duration"
DIALOGUE_CUT, EXCHANGE_SPLIT, BOUNDARY = "dialogue_cut", "exchange_split", "boundary"
SCENE, HOOK, CLIFFHANGER, SEGMENTS = "scene", "hook", "cliffhanger", "segments"
HOOK_MODALITY = "hook_modality"
MODALITY_TYPES = {"visual": ("camera", "action"), "spoken": ("dialogue",), "audio": ("sfx", "dialogue")}


@dataclass
class Result:
    ok: bool                                                # status != "reject"
    status: str = "reject"                                  # pass | review | reject
    reasons: list[str] = field(default_factory=list)       # reject codes
    details: list[str] = field(default_factory=list)       # human-readable
    warnings: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)        # review codes (rankable, but a human must look)
    review_details: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    duration: float | None = None
    segments: list[dict] = field(default_factory=list)
    hook_delay: float | None = None
    hook_modality: str | None = None


def _union_gaps(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    """Gaps in [lo, hi] not covered by any interval (including leading/trailing)."""
    gaps, cur = [], lo
    for a, b in sorted(intervals):
        if a > cur + EPS:
            gaps.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi - EPS:
        gaps.append((cur, hi))
    return [(a, b) for a, b in gaps if b - a > EPS]


def validate(ep, cand: dict, rules: dict | None = None, *, require_hook: bool = True) -> Result:
    r = {**DEFAULT_RULES, **(rules or {})}
    res = Result(ok=False)

    def fail(code: str, msg: str) -> None:
        if code not in res.reasons:
            res.reasons.append(code)
        res.details.append(msg)

    units = ep.units
    index = {u.id: i for i, u in enumerate(units)}

    # --- multi-segment: schema carries segments[], generation/validation is continuous-only in P0
    if cand.get("segments") and len(cand["segments"]) != 1:
        fail(SEGMENTS, "only single-segment candidates are supported in P0")
        return res

    # --- 1. IDs exist
    hook_ids = list(cand.get("hook_units") or ([cand["hook_unit"]] if cand.get("hook_unit") else []))
    for key in ("start_unit", "end_unit"):
        if cand.get(key) not in index:
            fail(IDS, f"{key} {cand.get(key)!r} not found in episode")
    if require_hook and not hook_ids:
        fail(IDS, "no hook unit given")
    for h in hook_ids:
        if h not in index:
            fail(IDS, f"hook unit {h!r} not found in episode")
    if cand.get("cliffhanger_unit") is not None and cand["cliffhanger_unit"] not in index:
        fail(IDS, f"cliffhanger_unit {cand['cliffhanger_unit']!r} not found in episode")
    if res.reasons:
        return res

    a, b = index[cand["start_unit"]], index[cand["end_unit"]]

    # --- 2. order
    if a > b:
        fail(ORDER, f"start_unit {cand['start_unit']} comes after end_unit {cand['end_unit']}")
        return res
    window = units[a:b + 1]
    win_ids = {u.id for u in window}

    # --- 3. timestamps valid, non-negative, ordered
    for u in window:
        if u.start is None or u.end is None:
            fail(TIMESTAMP, f"{u.id} has missing/malformed timestamp")
        elif u.start < 0 or u.end < 0 or u.end < u.start - EPS:
            fail(TIMESTAMP, f"{u.id} has invalid times {u.start}..{u.end}")
    if TIMESTAMP in res.reasons:
        return res

    cs = min(u.start for u in window)
    ce = max(u.end for u in window)
    dur = round(ce - cs, 3)
    res.start, res.end, res.duration = cs, ce, dur
    res.segments = [{"start_unit": cand["start_unit"], "end_unit": cand["end_unit"],
                     "start": cs, "end": ce}]

    # --- 4. duration (inclusive)
    if dur < r["min_clip_s"] - EPS:
        fail(DURATION, f"{dur}s is under the {r['min_clip_s']}s minimum")
    elif dur > r["max_clip_s"] + EPS:
        fail(DURATION, f"{dur}s is over the {r['max_clip_s']}s maximum")

    # --- 5. dialogue: no cut mid-line, no split exchange
    for u in units:
        if u.type != "dialogue" or u.start is None or u.end is None:
            continue
        inside_window = u.id in win_ids
        for t, name in ((cs, "start"), (ce, "end")):
            if u.start + EPS < t < u.end - EPS and not inside_window:
                fail(DIALOGUE_CUT, f"clip {name} {t}s cuts through {u.id} ({u.start}-{u.end}): "
                                   f"\"{u.text[:40]}\"")
        # spoken line that plays inside the clip but was left out of the window
        if not inside_window and u.start < ce - EPS and u.end > cs + EPS and not (
                u.start + EPS < cs < u.end - EPS or u.start + EPS < ce < u.end - EPS):
            fail(BOUNDARY, f"{u.id} (dialogue {u.start}-{u.end}) plays inside the clip but is outside the window")
    members: dict[str, list] = {}
    for u in units:
        if u.exchange:
            members.setdefault(u.exchange, []).append(u)
    for ex, ms in members.items():
        n_in = sum(m.id in win_ids for m in ms)
        if 0 < n_in < len(ms):
            fail(EXCHANGE_SPLIT, f"exchange {ex} ({', '.join(m.id for m in ms)}) is only partly inside the clip")

    # --- 6. boundary: speech left out but playing inside the clip is fatal (handled in 5). An action/sfx unit
    # from outside the window that overlaps the clip is a review flag (sound/action can be trimmed in the edit);
    # a straddling camera shot is a warning only.
    for u in units:
        if u.id in win_ids or u.start is None or u.end is None or u.type in ("dialogue",):
            continue
        overlaps = u.start < ce - EPS and u.end > cs + EPS
        if not overlaps:
            continue
        if u.type == "camera":
            res.warnings.append(f"camera shot {u.id} ({u.start}-{u.end}) straddles a cut")
        else:
            if "straddled_action_or_sfx" not in res.review:
                res.review.append("straddled_action_or_sfx")
            res.review_details.append(f"{u.type} {u.id} ({u.start}-{u.end}) overlaps the clip from outside the "
                                      f"window: check the cut on footage")

    # --- 7. scene
    scenes = {u.scene for u in window}
    if len(scenes) > 1 and not cand.get("allow_cross_scene"):
        fail(SCENE, f"window crosses scenes {sorted(map(str, scenes))}")

    # --- 8. timeline coverage. Never a reject: gaps are flagged (full coverage) or noted (dialogue-only source).
    def spans(pred):
        return [(max(u.start, cs), min(u.end, ce)) for u in units
                if pred(u) and u.start is not None and u.end is not None and u.start < ce and u.end > cs]
    coverage = getattr(ep, "visual_coverage", "full")
    for g0, g1 in _union_gaps(spans(lambda u: True), cs, ce):
        gap = round(g1 - g0, 2)
        if gap <= r["dead_air_review_s"] + EPS:
            continue
        if coverage == "full":
            res.review.append("uncovered_gap")
            res.review_details.append(f"{gap}s with nothing scripted ({g0}-{g1}): check the footage")
        else:
            res.warnings.append(f"unobserved gap {gap}s ({g0}-{g1}): {coverage} source, visual content unknown")
    if coverage == "full":  # informational: long visual-only stretches (scars reveal, fire...) are often the drama
        for g0, g1 in _union_gaps(spans(lambda u: u.type != "camera"), cs, ce):
            if g1 - g0 > r["dead_air_review_s"] + EPS:
                res.warnings.append(f"camera-only stretch {round(g1 - g0, 2)}s ({g0}-{g1})")

    # --- 9. hook: every hook unit inside the clip; the earliest begins within N s; modality matches the units
    if hook_ids:
        hs = [units[index[h]] for h in hook_ids]
        outside = [h.id for h in hs if h.id not in win_ids]
        if outside:
            fail(HOOK, f"hook unit(s) {', '.join(outside)} outside the window")
        else:
            first = min(hs, key=lambda u: u.start)
            res.hook_delay = round(first.start - cs, 3)
            if res.hook_delay > r["hook_start_within_s"] + EPS:
                fail(HOOK, f"hook {first.id} begins {round(res.hook_delay, 2)}s after clip start "
                           f"(limit {r['hook_start_within_s']}s)")
        mod = cand.get("hook_modality")
        if mod is not None:
            res.hook_modality = mod
            if mod not in MODALITY_TYPES:
                fail(HOOK_MODALITY, f"hook modality {mod!r} is not visual/spoken/audio")
            elif not any(h.type in MODALITY_TYPES[mod] for h in hs):
                fail(HOOK_MODALITY, f"{mod} hook claimed but hook units are {sorted({h.type for h in hs})}")

    # --- 10. cliffhanger evidence in the final fraction of the clip
    if cand.get("cliffhanger_unit") in index:
        c = units[index[cand["cliffhanger_unit"]]]
        limit = cs + (1 - r["cliffhanger_final_fraction"]) * dur
        if c.id not in win_ids:
            fail(CLIFFHANGER, f"cliffhanger unit {c.id} is outside the window")
        elif c.start < limit - EPS:
            fail(CLIFFHANGER, f"cliffhanger {c.id} starts at {c.start}s, before the final "
                              f"{int(r['cliffhanger_final_fraction'] * 100)}% of the clip ({round(limit, 2)}s)")

    res.ok = not res.reasons
    res.status = "reject" if res.reasons else ("review" if res.review else "pass")
    return res
