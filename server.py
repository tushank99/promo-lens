"""PromoLens web API + static frontend. Stdlib only.

  python server.py            # http://localhost:8000

Endpoints:
  GET  /api/health            -> {"ok": true, "keys": {...}}
  POST /api/run               {"filename": "ep1.txt", "text": "...", "format": "auto"} -> {"job_id": "..."}
  GET  /api/jobs/<id>         -> {"status": "queued|running|done|error", "result": {...}, "error": "..."}
Everything else is served from web/.
"""
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from promolense.adapters import load_episode  # noqa: E402
from promolense.llm import load_config, load_env  # noqa: E402
from promolense.pipeline import run_episode  # noqa: E402

load_env()
WEB = ROOT / "web"
MAX_BYTES = 2_000_000
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()  # one pipeline at a time: provider pacing/rate limits are process-wide


def _excerpt(ep, start_id, end_id):
    ids = [u.id for u in ep.units]
    try:
        a, b = ids.index(start_id), ids.index(end_id)
    except ValueError:
        return []
    return [{"id": u.id, "type": u.type, "speaker": u.speaker, "start": u.start, "text": u.text}
            for u in ep.units[a:b + 1]]


def _run_job(job_id: str, filename: str, text: str, fmt: str):
    job = JOBS[job_id]
    with RUN_LOCK:
        job["status"] = "running"
        job["started"] = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                safe = "".join(c for c in Path(filename).name if c.isalnum() or c in "._- ") or "episode.txt"
                src = Path(tmp) / safe
                src.write_text(text, encoding="utf-8")
                ep = load_episode(src, fmt)
                if ep.rejected:
                    raise ValueError(f"Script rejected: {ep.reject_reason}")
                final = run_episode(ep, Path(tmp) / "out", load_config())
                for k in final.get("final", []):
                    k["excerpt"] = _excerpt(ep, k["start_unit_id"], k["end_unit_id"])
                final.pop("all_candidates", None)
                final["source"] = {"file": safe, "units": len(ep.units), "scenes": len(ep.scenes)}
                job["result"] = final
                job["status"] = "done"
        except Exception as e:  # report to the client, keep the server alive
            traceback.print_exc()
            job["status"] = "error"
            job["error"] = str(e) or e.__class__.__name__
        job["finished"] = time.time()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def _json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self._json(200, {"ok": True, "keys": {k: bool(os.environ.get(k)) for k in
                                                         ("GEMINI_API_KEY", "GROQ_API_KEY")}})
        if self.path.startswith("/api/jobs/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1])
            if not job:
                return self._json(404, {"error": "unknown job"})
            out = {k: v for k, v in job.items() if k != "text"}
            if job["status"] == "queued":
                out["position"] = sum(1 for j in JOBS.values() if j["status"] == "queued"
                                      and j["created"] <= job["created"])
            return self._json(200, out)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/run"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self._json(413, {"error": "empty or too large (max 2 MB)"})
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            text = body["text"]
        except (ValueError, KeyError, TypeError):
            return self._json(400, {"error": "send JSON {filename, text, format}"})
        if not isinstance(text, str) or not text.strip():
            return self._json(400, {"error": "script text is empty"})
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"id": job_id, "status": "queued", "created": time.time(),
                        "filename": body.get("filename") or "episode.txt"}
        threading.Thread(target=_run_job, daemon=True,
                         args=(job_id, JOBS[job_id]["filename"], text, body.get("format") or "auto")).start()
        return self._json(202, {"job_id": job_id})


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"PromoLens on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
