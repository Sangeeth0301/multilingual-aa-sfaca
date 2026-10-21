# SFA-CA backend API

FastAPI server in `api.py`. It loads the model once, serves the web page from `../web/` at `/`, and exposes:

| Method | Path | Description |
|:-|:-|:-|
| GET | `/health` | Whether the model is loaded and the class names |
| POST | `/api/analyze` | Form field `text`, or an uploaded `file` (.txt .md .docx .pdf). Returns the analysis JSON |
| POST | `/api/export` | Same input; returns a highlighted HTML report |

Settings (environment variables): `ADAPTER_DIR` (trained adapters), `SFACA_STUB=1` (placeholder scorer),
`ALLOWED_ORIGINS` (comma-separated CORS origins, default `*`), `SFACA_MAX_WINDOWS` (request cap, default 400), `PORT` (default 7860).

## Examples

```bash
curl http://localhost:7860/health
curl -F "text=Your text of at least forty characters goes here, as a couple of sentences." http://localhost:7860/api/analyze
curl -F "file=@essay.docx" http://localhost:7860/api/analyze
curl -F "file=@essay.pdf" http://localhost:7860/api/export -o report.html
```

Errors are JSON: `{"detail": "..."}` with status 400 (too short), 413 (too large), 415 (wrong type), 422 (unreadable or
scanned file), 503 (model not loaded). Full description: `../REPORT.md`.

`app.py` is the earlier Gradio dashboard, kept for reference; it is not the deployed interface.
