"""Input adapters → canonical Episode. The core never sees source formats.

  load_episode(path, fmt="auto")      screenplay_txt | subtitles (.srt/.vtt) | canonical_json
  build_manifest(paths)               source file ↔ header episode ↔ content hash ↔ status (canonical/duplicate)

Adapters never invent information: subtitles have no camera/action units, so they declare
visual_coverage="dialogue_only" and the validator treats gaps as unobserved, not dead air.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .parser import parse_file, parse_ts
from .schema import BAD_FRACTION_DEFAULT, Episode, Scene, Unit, finalize_episode, from_canonical

FORMATS = ("screenplay_txt", "subtitles", "canonical_json")


def detect_format(path: Path, head: str) -> str:
    ext = path.suffix.lower()
    if ext in (".srt", ".vtt"):
        return "subtitles"
    if ext == ".json":
        return "canonical_json"
    if head.lstrip().startswith("WEBVTT") or re.search(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->", head):
        return "subtitles"
    if head.lstrip().startswith("{"):
        return "canonical_json"
    return "screenplay_txt"


def load_episode(path: str | Path, fmt: str = "auto", *, bad_fraction: float = BAD_FRACTION_DEFAULT) -> Episode:
    p = Path(path)
    raw = p.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    fmt = detect_format(p, text[:2000]) if fmt == "auto" else fmt
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; use one of {FORMATS}")
    sha = hashlib.sha256(raw).hexdigest()
    if fmt == "screenplay_txt":
        ep = parse_file(p, bad_fraction=bad_fraction)
    elif fmt == "subtitles":
        ep = parse_subtitles(text, number=_filename_number(p), bad_fraction=bad_fraction)
    else:
        try:
            ep = from_canonical(json.loads(text), bad_fraction=bad_fraction)
        except (json.JSONDecodeError, ValueError) as e:
            ep = finalize_episode(Episode(_filename_number(p), None, None, [], [],
                                          issues=[f"invalid canonical JSON: {e}"]), bad_fraction)
    ep.source = {**ep.source, "file": p.name, "format": fmt, "sha256": sha,
                 "adapter": ep.source.get("adapter", fmt)}
    return ep


def _filename_number(p: Path) -> int | None:
    m = re.search(r"(?:episode|ep)[-_ ]?0*(\d+)", p.stem, re.I)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ SRT / VTT
_CUE_TIME = re.compile(r"^\s*(?P<s>[\d:.,]+)\s*-->\s*(?P<e>[\d:.,]+)")
_VOICE = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>")
_TAG = re.compile(r"<[^>]+>")
_SPEAKER = re.compile(r"^([A-Z][A-Z .'\-]{1,30}):\s+(.*)$")


def parse_subtitles(text: str, *, number: int | None = None, bad_fraction: float = BAD_FRACTION_DEFAULT) -> Episode:
    """SRT and WebVTT. One dialogue unit per cue. No scenes in subtitles → one scene; no visual units."""
    lines = text.replace("\r\n", "\n").split("\n")
    ep = Episode(number, None, None, [Scene("01", "(subtitles: no scene information)")], [],
                 visual_coverage="dialogue_only")
    ep.issues.append("subtitle source: no scene headings (one scene assumed, cross-scene check is inactive), "
                     "no camera/action units (visual content unknown), no logline")
    tag = f"e{number:02d}" if number is not None else "e00"
    i, n = 0, len(lines)
    while i < n:
        m = _CUE_TIME.match(lines[i])
        if not m:
            i += 1
            continue
        start, end = parse_ts(_norm_ts(m.group("s"))), parse_ts(_norm_ts(m.group("e")))
        i += 1
        body = []
        while i < n and lines[i].strip() and not _CUE_TIME.match(lines[i]):
            body.append(lines[i].strip())
            i += 1
        if body and re.fullmatch(r"\d+", body[-1]) and i < n and _CUE_TIME.match(lines[i]):
            body.pop()  # SRT index of the next cue when the blank line is missing
        speaker = None
        joined = " ".join(body)
        v = _VOICE.search(joined)
        if v:
            speaker = v.group(1).strip()
        joined = _TAG.sub("", joined).strip()
        sm = _SPEAKER.match(joined)
        if sm and not speaker:
            speaker, joined = sm.group(1).strip(), sm.group(2).strip()
        flags = []
        if start is None or end is None:
            flags.append("bad_timestamp")
            start = end = None  # never keep half a timestamp (same rule as the screenplay parser)
        elif end < start:
            flags.append("bad_order")
        if not joined:
            flags.append("missing_text")
        uid = f"{tag}_u{len(ep.units) + 1:03d}"
        ep.units.append(Unit(uid, "01", "dialogue", speaker, None, start, end, joined, flags,
                             source_type="subtitle", timing_confidence="exact"))
    timed = [u for u in ep.units if u.start is not None]
    if timed:
        ep.scenes[0].start, ep.scenes[0].end = min(u.start for u in timed), max(u.end for u in timed)
    ep.source = {"adapter": "subtitles"}
    return finalize_episode(ep, bad_fraction)


def _norm_ts(s: str) -> str:
    """SRT 00:00:01,500 and VTT 00:01.500 are already forms parse_ts accepts."""
    return s.strip()


# ------------------------------------------------------------------ manifest
def build_manifest(paths: list[Path], fmt: str = "auto") -> list[dict]:
    """One row per source file. Exact duplicates are kept in the manifest but marked, never silently dropped."""
    rows, first_by_hash, first_by_episode = [], {}, {}
    for p in sorted(paths):
        try:
            ep = load_episode(p, fmt)
        except Exception as e:  # unreadable file: recorded, not fatal
            rows.append({"source_file": p.name, "path": str(p), "status": "error", "error": repr(e)})
            continue
        sha = ep.source["sha256"]
        row = {"source_file": p.name, "path": str(p), "format": ep.source["format"],
               "filename_number": _filename_number(p), "header_episode": ep.number, "episode_id": ep.episode_id,
               "content_sha256": sha, "units": len(ep.units), "scenes": len(ep.scenes),
               "visual_coverage": ep.visual_coverage, "parse_rejected": ep.rejected,
               "reject_reason": ep.reject_reason, "warnings": len(ep.issues) + sum(bool(u.flags) for u in ep.units)}
        if sha in first_by_hash:
            row["status"] = f"duplicate_of:{first_by_hash[sha]}"
        elif ep.rejected:
            row["status"] = "rejected"
        elif ep.number is not None and ep.number in first_by_episode:
            row["status"] = f"conflict:same_header_as:{first_by_episode[ep.number]}"  # different content, same number
        else:
            row["status"] = "canonical"
        first_by_hash.setdefault(sha, p.name)
        if ep.number is not None and row["status"] == "canonical":
            first_by_episode.setdefault(ep.number, p.name)
        rows.append(row)
    return rows


def write_manifest(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ["source_file", "header_episode", "filename_number", "status", "content_sha256", "format",
            "units", "scenes", "visual_coverage", "warnings", "reject_reason"]
    with open(out_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
