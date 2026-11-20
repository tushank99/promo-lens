"""Swappable LLM layer: one OpenAI-compatible client, per-stage config, hash cache, call log.

complete_json(stage, prompt, schema_hint) -> dict
Swap provider/model by editing config.toml. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "outputs" / "logs" / "llm.jsonl"   # tests point this elsewhere
CACHE_DIR = ROOT / "cache"                            # tests point this elsewhere


def load_env(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (never overrides existing vars)."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for ln in p.read_text(encoding="utf-8-sig").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, _, v = ln.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.toml"
    with open(p, "rb") as f:
        return tomllib.load(f)


def stage_cfg(cfg: dict, stage: str, which: str = "provider") -> dict:
    """Resolve settings for a stage. which = 'provider' (primary) or 'fallback'.
    Layering: defaults < provider block < stage overrides (stage overrides apply to the primary only)."""
    out = dict(cfg["defaults"])
    st = dict(cfg.get("stages", {}).get(stage, {}))
    pname = st.pop("provider", None) if which == "provider" else None
    pname = pname or out.get(which)
    if pname:
        out.update(cfg.get("providers", {}).get(pname, {}))
    if which == "provider":
        out.update(st)   # stage overrides (model, pacing, fallback) win over the provider block
    out["_provider"] = pname
    return out


def _extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group())
        raise


def _http_chat(sc: dict, messages: list[dict]) -> tuple[str, dict]:
    key = os.environ.get(sc["api_key_env"], "")
    if not key:
        raise RuntimeError(f"Env var {sc['api_key_env']} is not set.")
    body = json.dumps({
        "model": sc["model"], "messages": messages,
        "temperature": sc.get("temperature", 0.0),
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        sc["base_url"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "promolense/0.1"})  # default urllib UA gets 403 from Groq/Cloudflare
    with urllib.request.urlopen(req, timeout=sc.get("timeout_s", 90)) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def fallback_name(cfg: dict, stage: str) -> str | None:
    """Fallback is per stage and explicit: stages.<stage>.fallback overrides defaults.fallback; "" disables it."""
    fb = cfg.get("stages", {}).get(stage, {}).get("fallback", cfg["defaults"].get("fallback"))
    return fb or None


def complete_json(stage: str, prompt: str, *, system: str = "Reply with a single valid JSON object only.",
                  cfg: dict | None = None, transport=None, use_cache: bool = True, meta: dict | None = None) -> dict:
    """transport(sc, messages) -> (text, usage). Injectable for tests.
    meta (optional dict) is filled with provenance: provider, model, cached, attempts, waited_s, secs, tokens."""
    meta = meta if meta is not None else {}
    cfg = cfg or load_config()
    sc = stage_cfg(cfg, stage)
    sc.pop("fallback", None)
    transport = transport or _http_chat
    fb_name = fallback_name(cfg, stage)
    try:
        return _complete(stage, prompt, system, sc, transport, use_cache, meta=meta)
    except Exception as primary_err:
        if not fb_name:
            raise
        fb = stage_cfg({**cfg, "defaults": {**cfg["defaults"], "fallback": fb_name}}, stage, "fallback")
        if fb.get("model") == sc.get("model") and fb.get("base_url") == sc.get("base_url"):
            raise
        _log(stage, sc, 0, 0, 0.0, f"fallback_to_{fb['_provider']}")
        meta["fallback_from"] = sc["_provider"]
        meta["primary_error"] = str(primary_err)[:300]
        try:
            return _complete(stage, prompt, system, fb, transport, use_cache, meta=meta)
        except Exception as fb_err:
            raise RuntimeError(f"primary ({sc['_provider']}): {primary_err}; fallback ({fb['_provider']}): {fb_err}")


# ---- pacing and backoff (simple, per provider, per process; enough for a 49-episode batch)
_sleep = time.sleep          # tests replace this
_now = time.monotonic
_last_call: dict[str, float] = {}
_pace_lock = threading.Lock()


def _pace(sc: dict) -> float:
    """Keep at least min_interval_s between requests to the same provider. Returns seconds waited."""
    gap = float(sc.get("min_interval_s", 0) or 0)
    if gap <= 0:
        return 0.0
    key = sc.get("_provider") or sc.get("base_url", "")
    with _pace_lock:
        wait = max(0.0, _last_call.get(key, -1e9) + gap - _now())
        if wait > 0:
            _sleep(wait)
        _last_call[key] = _now()
    return wait


_exhausted: dict[str, str] = {}   # provider -> quota message (reset by restarting the process)


def _retry_hint_s(body: str) -> float | None:
    """'Please try again in 7m12.5s' / 'retry in 12h16m31s' / 'in 45.2s' -> seconds."""
    m = re.search(r"(?:try again|retry) in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", body, re.I)
    if not m or not m.group(1):
        return None
    h = re.search(r"(\d+)h", m.group(1))
    mi = re.search(r"(\d+)m(?!s)", m.group(1))
    s = re.search(r"([\d.]+)s", m.group(1))
    return (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) + (float(s.group(1)) if s else 0)


QUOTA_WAIT_MAX_S = 900  # a rolling daily window that reopens within 15 min is waited out, not treated as exhausted


def _daily_quota_message(e: urllib.error.HTTPError) -> str | None:
    """A 429 that says a per-day quota is spent is not worth retrying, unless the provider says it reopens soon
    (Groq's tokens-per-day limit is a rolling window: 'try again in 7m12s'). Short hints are stored on the error."""
    try:
        body = e.read().decode("utf-8", "replace")
    except Exception:
        body = ""
    e.body_text = body
    hint = _retry_hint_s(body)
    e.retry_hint_s = hint
    daily = re.search(r"per ?day|PerDay|requests_per_day|\(TPD\)|\(RPD\)", body, re.I)
    if hint is not None and hint <= QUOTA_WAIT_MAX_S:
        return None                       # wait it out (see _backoff)
    if daily or (hint is not None and hint > QUOTA_WAIT_MAX_S):
        msg = re.search(r"(Quota exceeded for metric|Rate limit reached for model)[^\n\\]*", body)
        return (msg.group(0) if msg else "daily quota exceeded")[:300]
    return None


def _backoff(sc: dict, attempt: int, err: Exception | None) -> float:
    hint = getattr(err, "retry_hint_s", None)
    if hint is not None and hint <= QUOTA_WAIT_MAX_S:   # provider told us exactly when the window reopens
        return hint + 2.0
    ra = None
    hdrs = getattr(err, "headers", None)
    if hdrs is not None:
        try:
            ra = float(hdrs.get("Retry-After"))
        except (TypeError, ValueError):
            ra = None
    if ra is not None and ra >= 0:
        return min(ra, float(sc.get("max_backoff_s", 60)))
    base = float(sc.get("backoff_base_s", 2.0))
    return min(base * (2 ** attempt), float(sc.get("max_backoff_s", 60))) * (0.5 + random.random() * 0.5)


def _complete(stage, prompt, system, sc, transport, use_cache, meta=None):
    meta = meta if meta is not None else {}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    # cache key = the provider+model that actually answers. (Before: a fallback answer was stored under the
    # primary's key and later served as if the primary had produced it.)
    h = hashlib.sha256(json.dumps([sc.get("_provider"), sc["model"], sc.get("temperature"), messages]).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{h}.json"
    if use_cache and cache_file.exists():
        blob = json.loads(cache_file.read_text(encoding="utf-8"))
        bm = blob.get("_meta", {}) if isinstance(blob, dict) else {}
        if "result" in blob and bm.get("provider") == sc.get("_provider") and bm.get("model") == sc["model"]:
            _log(stage, sc, 0, 0, 0.0, "cache_hit")
            meta.update(bm, cached=True)
            return blob["result"]
    if sc.get("_provider") in _exhausted:  # circuit breaker: daily quota already spent in this process
        meta.update(provider=sc.get("_provider"), model=sc["model"], failed=True, quota_exhausted=True)
        raise RuntimeError(f"quota exhausted for {sc.get('_provider')}/{sc['model']} (not retried): "
                           f"{_exhausted[sc.get('_provider')]}")

    last_err: Exception | None = None
    repaired = False
    waited = 0.0
    max_wait = float(sc.get("max_total_wait_s", 180))
    for attempt in range(int(sc.get("max_retries", 4))):
        waited += _pace(sc)
        t0 = time.time()
        try:
            text, usage = transport(sc, messages)
            try:
                result = _extract_json(text)
            except json.JSONDecodeError as e:
                if repaired:
                    raise
                repaired = True  # one repair retry
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": "That was not valid JSON. Return only the JSON object."}]
                last_err = e
                continue
            _log(stage, sc, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                 time.time() - t0, "ok")
            meta.update(provider=sc.get("_provider"), model=sc["model"], cached=False, attempts=attempt + 1,
                        waited_s=round(waited, 1), secs=round(time.time() - t0, 2),
                        tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0))
            if use_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"_meta": {k: meta[k] for k in ("provider", "model")},
                                                  "result": result}, ensure_ascii=False), encoding="utf-8")
            return result
        except urllib.error.HTTPError as e:
            last_err = e
            _log(stage, sc, 0, 0, time.time() - t0, f"http_{e.code}")
            if e.code == 429:
                quota = _daily_quota_message(e)
                if quota:  # waiting cannot help: fail fast and stop calling this provider for the rest of the run
                    _exhausted[sc.get("_provider")] = quota
                    _log(stage, sc, 0, 0, 0.0, "quota_exhausted")
                    meta.update(provider=sc.get("_provider"), model=sc["model"], failed=True, quota_exhausted=True)
                    raise RuntimeError(f"quota exhausted for {sc.get('_provider')}/{sc['model']}: {quota}")
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            _log(stage, sc, 0, 0, time.time() - t0, "network")
        if attempt + 1 >= int(sc.get("max_retries", 4)):
            break
        w = _backoff(sc, attempt, last_err)
        hinted = getattr(last_err, "retry_hint_s", None) is not None
        if waited + w > max_wait and not (hinted and waited == 0):
            _log(stage, sc, 0, 0, 0.0, "wait_budget_exhausted")
            break
        _sleep(w)
        waited += w
    _log(stage, sc, 0, 0, 0.0, "failed")
    meta.update(provider=sc.get("_provider"), model=sc["model"], waited_s=round(waited, 1), failed=True)
    raise RuntimeError(f"LLM call failed for stage '{stage}' ({sc.get('_provider')}/{sc['model']}): {last_err}")


def _log(stage: str, sc: dict, tin: int, tout: int, secs: float, status: str) -> None:
    p = LOG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "stage": stage, "model": sc["model"], "provider": sc.get("_provider"),
                            "tokens_in": tin, "tokens_out": tout, "secs": round(secs, 2),
                            "status": status}) + "\n")
