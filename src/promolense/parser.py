"""Parse a Kuku microdrama screenplay into timestamped units. Pure code, no LLM.

Never estimates timestamps: an episode without usable timing is rejected with a message.
"""
from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path

from .schema import BAD_FRACTION_DEFAULT, Episode, Scene, Unit, finalize_episode  # noqa: F401  (re-export)

# 1:02 | 01:02 | 1:02.4 | 01:02.30 | 00:01:02 | 00:01:02.5   (comma decimal tolerated)
_TS = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d+))?$")
_HEADER = re.compile(r"^\*\*(?P<label>.+?):\*\*\s*$")
_TIME_LINE = re.compile(r"^Start:\s*(?P<s>.*?)\s*\|\s*End:\s*(?P<e>.*?)\s*$", re.I)
_SCENE = re.compile(r"^SCENE\s+(?P<n>\w+)\s*$", re.I)
_PAREN = re.compile(r"\(([^)]*)\)")
_MODIFIERS = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONT’D"}

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def parse_ts(raw: str | None) -> float | None:
    """Return seconds, or None if missing/malformed."""
    if raw is None:
        return None
    s = raw.strip().strip("()").strip()
    m = _TS.match(s)
    if not m:
        return None
    h, mi, sec, frac = m.groups()
    if int(sec) >= 60 or (h is not None and int(mi) >= 60):
        return None
    total = int(h or 0) * 3600 + int(mi) * 60 + int(sec)
    if frac:
        total += float("0." + frac)
    return round(total, 3)


def _split_label(label: str) -> tuple[str, str | None, str, list[str]]:
    """'NURSE (O.S.) (Firm)' -> ('NURSE', 'Firm', type-key, ['O.S.'])."""
    name = _PAREN.sub("", label).strip()
    groups = [g.strip() for g in _PAREN.findall(label)]
    mods = [g for g in groups if g.upper() in _MODIFIERS]
    non_mod = [g for g in groups if g.upper() not in _MODIFIERS]
    tone = non_mod[-1] if non_mod else None
    return name, tone, name.upper(), mods


def _header_value(lines: list[str], key: str) -> str | None:
    for ln in lines:
        if ln.upper().startswith(key):
            return ln.split(":", 1)[1].strip() or None
    return None


def parse_text(text: str, *, fallback_number: int | None = None,
               bad_fraction: float = BAD_FRACTION_DEFAULT) -> Episode:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    # --- header (before SCRIPT START) ---
    try:
        script_at = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("--- SCRIPT START"))
    except StopIteration:
        script_at = 0
    head = lines[:script_at]
    num_raw = _header_value(head, "EPISODE NUMBER")
    number = fallback_number
    if num_raw:
        m = re.search(r"\d+", num_raw)
        if m:
            number = int(m.group())
    logline = _header_value(head, "EPISODE LOGLINE")
    continuity = _header_value(head, "CONTINUITY NOTES")
    ep = Episode(number, logline, continuity, [], [])
    if not logline:
        ep.issues.append("missing logline")
    ep_tag = f"e{(number or 0):02d}"

    body = lines[script_at:]
    scenes: dict[str, Scene] = {}
    cur_scene: Scene | None = None
    pending: dict | None = None  # unit under construction

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        p = pending
        pending = None
        flags = p["flags"]
        text_ = " ".join(p["text"]).strip()
        if len(text_) >= 2 and text_[0] == '"' and text_[-1] == '"':
            text_ = text_[1:-1].strip()
        if not text_:
            flags.append("missing_text")
        if p["start"] is None and "bad_timestamp" not in flags and "missing_timestamp" not in flags:
            flags.append("missing_timestamp")
        if p["start"] is not None and p["end"] is not None and p["end"] < p["start"]:
            flags.append("bad_order")
        if cur_scene is None:
            flags.append("no_scene")
        uid = f"{ep_tag}_u{len(ep.units) + 1:03d}"
        ep.units.append(Unit(uid, p["scene"], p["type"], p["speaker"], p["tone"],
                             p["start"], p["end"], text_, flags, modifiers=p["mods"]))

    for ln in body:
        s = ln.strip()
        if not s or s.upper().startswith("--- SCRIPT"):
            if s.upper().startswith("--- SCRIPT END"):
                flush()
            continue
        sm = _SCENE.match(s)
        if sm:
            flush()
            sid = sm.group("n")
            cur_scene = scenes.setdefault(sid, Scene(sid))
            continue
        if cur_scene is not None and pending is None and ":" in s:
            k, _, v = s.partition(":")
            kl = k.strip().lower()
            if kl == "scene heading":
                cur_scene.heading = v.strip(); continue
            if kl == "scene start":
                cur_scene.start = parse_ts(v); continue
            if kl == "scene end":
                cur_scene.end = parse_ts(v); continue
            if kl == "actions and dialogue":
                continue
        hm = _HEADER.match(s)
        if hm:
            flush()
            name, tone, key, mods = _split_label(hm.group("label"))
            utype = NON_DIALOGUE.get(key, "dialogue")
            pending = dict(type=utype, speaker=name if utype == "dialogue" else None,
                           tone=tone, mods=mods, start=None, end=None, text=[], flags=[],
                           scene=cur_scene.id if cur_scene else None)
            if utype == "dialogue" and tone is None:
                pending["flags"].append("missing_tone")  # informational only
            continue
        if pending is not None:
            tm = _TIME_LINE.match(s)
            if tm and pending["start"] is None and "bad_timestamp" not in pending["flags"]:
                a, b = parse_ts(tm.group("s")), parse_ts(tm.group("e"))
                if a is None or b is None:
                    pending["flags"].append("bad_timestamp")
                else:
                    pending["start"], pending["end"] = a, b
                continue
            pending["text"].append(s)
    flush()

    ep.scenes = list(scenes.values())
    ep.visual_coverage = "full"
    return finalize_episode(ep, bad_fraction)


def parse_file(path: str | Path, **kw) -> Episode:
    """Screenplay TXT adapter entry point. The episode number comes from the header (authoritative);
    the filename number is only a fallback when the header has none."""
    p = Path(path)
    fallback = None
    m = re.search(r"episode-(\d+)", p.name)
    if m:
        fallback = int(m.group(1))
    raw = p.read_bytes()
    ep = parse_text(raw.decode("utf-8-sig"), fallback_number=fallback, **kw)
    ep.source = {"file": p.name, "format": "screenplay_txt", "adapter": "screenplay",
                 "sha256": hashlib.sha256(raw).hexdigest(), "filename_number": fallback}
    if fallback is not None and ep.number is not None and fallback != ep.number:
        ep.issues.append(f"filename says episode {fallback}, header says {ep.number}; header used")
    return ep
