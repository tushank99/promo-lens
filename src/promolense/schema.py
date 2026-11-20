"""Canonical episode format: the only thing the PromoLens core consumes.

Every input adapter (screenplay TXT, SRT/VTT subtitles, canonical JSON) produces an `Episode`, then calls
`finalize_episode()` so integrity checks, overlap groups and rejection rules are identical for all sources.

Canonical JSON (nested, for import/export/preview):
{
  "schema_version": "1",
  "episode_id": "ep01", "episode_number": 1,
  "visual_coverage": "full" | "dialogue_only",
  "metadata": {"logline": ..., "continuity": ...},
  "source": {"file": ..., "format": ..., "sha256": ..., "adapter": ...},
  "scenes": [{"scene_id": "01", "heading": ..., "start": 0.0, "end": 138.0,
              "units": [{"unit_id": "e01_u001", "start": 0.0, "end": 4.8, "unit_type": "camera", "text": ...,
                         "speaker": null, "tone": null, "modifiers": [], "source_type": "screenplay",
                         "timing_confidence": "exact", "overlap_group": null, "flags": [], "metadata": {}}]}],
  "issues": [], "rejected": false, "reject_reason": null
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "1"
UNIT_TYPES = ("dialogue", "camera", "action", "sfx")
COVERAGE = ("full", "dialogue_only")
BAD_FRACTION_DEFAULT = 0.02


@dataclass
class Unit:
    id: str
    scene: str | None
    type: str  # dialogue | camera | action | sfx
    speaker: str | None
    tone: str | None
    start: float | None
    end: float | None
    text: str
    flags: list[str] = field(default_factory=list)
    exchange: str | None = None              # overlap_group: dialogue-on-dialogue overlap
    modifiers: list[str] = field(default_factory=list)   # e.g. V.O., O.S.
    source_type: str = "screenplay"          # screenplay | subtitle | json
    timing_confidence: str = "exact"         # exact | estimated
    metadata: dict = field(default_factory=dict)

    # flags that make a unit unusable; everything else (missing_tone, outside_scene_range, no_scene) is informational
    BAD_FLAGS = ("bad_timestamp", "bad_order", "missing_timestamp", "missing_text", "bad_type")

    @property
    def bad(self) -> bool:
        return any(f in self.BAD_FLAGS for f in self.flags)


@dataclass
class Scene:
    id: str
    heading: str | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class Episode:
    number: int | None
    logline: str | None
    continuity: str | None
    scenes: list[Scene]
    units: list[Unit]
    issues: list[str] = field(default_factory=list)
    rejected: bool = False
    reject_reason: str | None = None
    episode_id: str | None = None
    visual_coverage: str = "full"            # full | dialogue_only (subtitles: visuals unknown)
    source: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ shared integrity checks
def finalize_episode(ep: Episode, bad_fraction: float = BAD_FRACTION_DEFAULT) -> Episode:
    if ep.episode_id is None:
        ep.episode_id = f"ep{ep.number:02d}" if ep.number is not None else "ep_unknown"
    if ep.visual_coverage not in COVERAGE:
        ep.issues.append(f"unknown visual_coverage {ep.visual_coverage!r}; treated as dialogue_only")
        ep.visual_coverage = "dialogue_only"
    _mark_scene_range_issues(ep)
    _mark_exchanges(ep)
    _decide_rejection(ep, bad_fraction)
    return ep


def _mark_scene_range_issues(ep: Episode) -> None:
    by_id = {s.id: s for s in ep.scenes}
    for s in ep.scenes:
        if s.start is None or s.end is None:
            ep.issues.append(f"scene {s.id}: missing start/end")
    for u in ep.units:
        sc = by_id.get(u.scene) if u.scene else None
        if sc and sc.start is not None and sc.end is not None and u.start is not None:
            if u.start < sc.start - 0.01 or (u.end or u.start) > sc.end + 0.01:
                if "outside_scene_range" not in u.flags:
                    u.flags.append("outside_scene_range")


def _mark_exchanges(ep: Episode) -> None:
    """Group dialogue units that overlap in time. No cut may fall inside a group."""
    d = sorted((u for u in ep.units if u.type == "dialogue" and u.start is not None and u.end is not None),
               key=lambda u: (u.start, u.end))
    group: list[Unit] = []
    gend = -1.0
    n = 0

    def close() -> None:
        nonlocal n
        if len(group) > 1:
            n += 1
            for g in group:
                g.exchange = f"x{n:02d}"

    for u in d:
        if group and u.start < gend - 1e-6:
            group.append(u)
            gend = max(gend, u.end)
        else:
            close()
            group = [u]
            gend = u.end
    close()


def _decide_rejection(ep: Episode, bad_fraction: float) -> None:
    total = len(ep.units)
    if total == 0:
        ep.rejected, ep.reject_reason = True, "No units found. Is this a timestamped script or subtitle file?"
        return
    timed = [u for u in ep.units if u.start is not None]
    if not timed:
        ep.rejected = True
        ep.reject_reason = "No timestamps in the input. Timestamps are never estimated; supply a timed script."
        return
    bad = [u for u in ep.units if u.bad]
    if len(bad) / total > bad_fraction:
        ep.rejected = True
        ep.reject_reason = (f"{len(bad)}/{total} units ({len(bad) / total:.1%}) have missing or "
                            f"malformed fields (limit {bad_fraction:.0%}): " + ", ".join(u.id for u in bad[:8]))


# ------------------------------------------------------------------ canonical JSON
def to_canonical(ep: Episode) -> dict:
    scenes = []
    order = [s.id for s in ep.scenes]
    by_scene: dict = {sid: [] for sid in order}
    for u in ep.units:
        by_scene.setdefault(u.scene, []).append(u)
    known = {s.id: s for s in ep.scenes}
    for sid, us in by_scene.items():
        s = known.get(sid, Scene(sid if sid is not None else "unassigned"))
        scenes.append({"scene_id": s.id, "heading": s.heading, "start": s.start, "end": s.end, "units": [
            {"unit_id": u.id, "start": u.start, "end": u.end, "unit_type": u.type, "text": u.text,
             "speaker": u.speaker, "tone": u.tone, "modifiers": u.modifiers, "source_type": u.source_type,
             "timing_confidence": u.timing_confidence, "overlap_group": u.exchange, "flags": u.flags,
             "metadata": u.metadata} for u in us]})
    return {"schema_version": SCHEMA_VERSION, "episode_id": ep.episode_id, "episode_number": ep.number,
            "visual_coverage": ep.visual_coverage,
            "metadata": {"logline": ep.logline, "continuity": ep.continuity, **ep.metadata},
            "source": ep.source, "scenes": scenes, "issues": ep.issues,
            "rejected": ep.rejected, "reject_reason": ep.reject_reason}


def from_canonical(d: dict, *, bad_fraction: float = BAD_FRACTION_DEFAULT) -> Episode:
    """Import canonical JSON. Required per unit: unit_id, start, end, text, unit_type. Missing → flagged, and the
    usual rejection rule applies. Overlap groups and scene-range flags are recomputed, not trusted."""
    if not isinstance(d, dict):
        raise ValueError("canonical JSON must be an object")
    meta = dict(d.get("metadata") or {})
    ep = Episode(number=d.get("episode_number"), logline=meta.pop("logline", None),
                 continuity=meta.pop("continuity", None), scenes=[], units=[],
                 episode_id=d.get("episode_id"), visual_coverage=d.get("visual_coverage", "full"),
                 source=dict(d.get("source") or {}), metadata=meta)
    if not d.get("episode_id"):
        ep.issues.append("missing episode_id")
    seen: set[str] = set()
    for i, sc in enumerate(d.get("scenes") or [], 1):
        sid = str(sc.get("scene_id") or f"s{i:02d}")
        if not sc.get("scene_id"):
            ep.issues.append(f"scene #{i}: missing scene_id, using {sid}")
        ep.scenes.append(Scene(sid, sc.get("heading"), _num(sc.get("start")), _num(sc.get("end"))))
        for j, ud in enumerate(sc.get("units") or [], 1):
            flags = [f for f in (ud.get("flags") or []) if f not in ("outside_scene_range",)]
            uid = ud.get("unit_id")
            if not uid or uid in seen:
                flags.append("missing_unit_id" if not uid else "duplicate_unit_id")
                uid = f"{ep.episode_id or 'ep'}_s{sid}_u{j:03d}"
            seen.add(uid)
            start, end = _num(ud.get("start")), _num(ud.get("end"))
            if start is None or end is None:
                flags.append("missing_timestamp")
            elif end < start:
                flags.append("bad_order")
            utype = ud.get("unit_type")
            if utype not in UNIT_TYPES:
                flags.append("bad_type")
            text = (ud.get("text") or "").strip()
            if not text:
                flags.append("missing_text")
            ep.units.append(Unit(uid, sid, utype if utype in UNIT_TYPES else "action", ud.get("speaker"),
                                 ud.get("tone"), start, end, text, flags, None, list(ud.get("modifiers") or []),
                                 ud.get("source_type", "json"), ud.get("timing_confidence", "exact"),
                                 dict(ud.get("metadata") or {})))
    return finalize_episode(ep, bad_fraction)


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        from .parser import parse_ts
        try:
            return float(x)
        except ValueError:
            return parse_ts(x)
    return None
