"""
api.py
------
FastAPI backend for SFA-CA.  Endpoints:
  GET  /health        model state
  POST /api/analyze   multipart: `text` or `file` (.txt .md .docx .pdf) -> analysis JSON
  POST /api/export    same input -> highlighted HTML report
The static frontend in ../web/ is served at / when present.

Env: ADAPTER_DIR (trained adapters), SFACA_STUB=1 (heuristic demo scorer),
     ALLOWED_ORIGINS (comma-separated CORS origins, default *).
"""
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))          # this member folder
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

from analyzer import analyze  # noqa: E402
from exporters import to_html  # noqa: E402
from file_parser import parse_uploaded_file  # noqa: E402

MAX_BYTES = 8 * 1024 * 1024
MAX_CHARS = 150_000
ALLOWED_EXT = {".txt", ".md", ".docx", ".pdf"}

_SCORER = None
_MODEL_NAME = None
_LOCK = threading.Lock()   # one forward pass at a time


def load_scorer():
    global _SCORER, _MODEL_NAME
    if _SCORER is not None:
        return
    if os.environ.get("SFACA_STUB") == "1":
        from stub_scorer import StubScorer
        _SCORER, _MODEL_NAME = StubScorer(), "stub"
        return
    from inference import InferenceEngine
    engine = InferenceEngine()
    engine.load_from_dir()
    _SCORER, _MODEL_NAME = engine, "sfa-ca"


@asynccontextmanager
async def lifespan(app):
    try:
        load_scorer()
    except Exception as e:  # keep the server up so /health explains the problem
        print(f"[SFA-CA] model not loaded: {e}")
    yield


app = FastAPI(title="SFA-CA AI-text attribution", version="2.0.0", lifespan=lifespan)
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok" if _SCORER is not None else "no_model", "model": _MODEL_NAME,
            "labels": list(_SCORER.labels) if _SCORER else []}


async def _read_input(text: Optional[str], file: Optional[UploadFile]):
    name = "pasted-text"
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"Unsupported file type '{ext}'. Use .txt, .md, .docx or .pdf.")
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File too large (max 8 MB).")
        name = file.filename
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "upload" + ext)
            with open(path, "wb") as f:
                f.write(data)
            try:
                text, _ = parse_uploaded_file(path)
            except Exception:
                raise HTTPException(422, "Could not read this file. It may be corrupt or password-protected.")
        if not text.strip():
            raise HTTPException(422, "No extractable text. Scanned/image-only PDFs are not supported (no OCR).")
    if not text or len(text.strip()) < 40:
        raise HTTPException(400, "Provide at least a couple of sentences (40+ characters).")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"Text too long (max {MAX_CHARS:,} characters).")
    return text, name


def _run(text: str) -> dict:
    if _SCORER is None:
        raise HTTPException(503, "Model not loaded. Set ADAPTER_DIR to the trained adapters (or SFACA_STUB=1 for a demo).")
    try:
        with _LOCK:
            res = analyze(text, _SCORER)
    except ValueError as e:
        raise HTTPException(413, str(e))
    except Exception:
        raise HTTPException(500, "Analysis failed.")
    res["model"] = _MODEL_NAME
    return res


@app.post("/api/analyze")
async def api_analyze(text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    text, name = await _read_input(text, file)
    res = _run(text)
    res["filename"] = name
    return res


@app.post("/api/export", response_class=HTMLResponse)
async def api_export(text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    text, name = await _read_input(text, file)
    return HTMLResponse(to_html(_run(text), name),
                        headers={"Content-Disposition": 'attachment; filename="sfaca-report.html"'})


_DOCS = os.path.join(HERE, "web")
if os.path.isdir(_DOCS):
    app.mount("/", StaticFiles(directory=_DOCS, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
