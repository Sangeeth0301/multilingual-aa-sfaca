# Member 4: Detection pipeline, backend, frontend and deployment

## 1. Role and objective

**Objective:** turn a trained model into a product that anyone can open with a link: paste text or upload a file, see how
likely it is AI-written, which sentences, and which generator.

This member owns everything between "a model that scores a short text" and "a web page a friend opens on another laptop":

1. **Pipeline:** read a document, find math and code, split it into sentences and windows, route by script, score,
   aggregate.
2. **Backend:** a web API that does this safely for uploaded files.
3. **Frontend:** the page that shows the result.
4. **Deployment:** how it is published and kept reachable.

## 2. What is in this folder

| File | Lines | Purpose |
|:-|:-:|:-|
| `src/analyzer.py` | 201 | The document pipeline: segment, window, route, calibrate, aggregate |
| `src/sanitizer.py` | 153 | Finds LaTeX and code spans |
| `src/router.py` | 50 | Detects the script of a text |
| `src/file_parser.py` | 88 | Extracts text from .txt, .md, .docx, .pdf |
| `src/exporters.py` | 40 | Builds the highlighted HTML report |
| `src/stub_scorer.py` | 30 | A placeholder scorer for tests and demos |
| `backend/api.py` | 139 | The FastAPI server |
| `backend/app.py` | 230 | The older Gradio dashboard (legacy, not deployed) |
| `web/index.html`, `config.js`, `api_url.txt` | 200 | The web page, its configuration and the live API address |
| `web/LOVABLE_PROMPT.md` | 45 | Specification for rebuilding the page in Lovable |
| `deploy/Dockerfile`, `SPACE_README.md`, `deploy_space.py` | 91 | Hugging Face Space bundle (ready, not used) |
| `ops/start_public.ps1`, `Start SFA-CA Public.bat` | 52 | One-click launcher: API, tunnel, publish URL |
| `tests/test_api.py` | 100 | API and pipeline tests |
| `PUSH_NOTES.md` | | Notes for the pull request |
| `../.github/workflows/pages.yml` | 33 | Publishes `web/` to GitHub Pages |

## 3. How the code works, file by file

### `src/analyzer.py`: the document pipeline
**Objective:** score a document of any length, even mixed human and AI, and say where the AI is.
**Steps in `analyze(text, scorer)`:**
1. **Segment.** `find_notation_spans` marks math and code. The remaining prose is split into sentences with a regular
   expression that also handles `。`, `！`, `？`, `؟` and newlines. The segments are **contiguous and keep exact character
   offsets**, so joining their texts reproduces the original, which is what lets the page highlight the right words.
2. **Window.** Sentences are grouped into overlapping windows of about 64 tokens (token count estimated cheaply: about 4
   characters per token, 1.3 for Chinese and Japanese), moving forward by half a window. Overlap means every sentence is
   judged in context twice.
3. **Route.** Each window goes to the adapter for its own script, so a Russian-and-English document uses two adapters.
4. **Score and calibrate.** `scorer.predict_probs(windows, family)` returns 8 probabilities. If `configs/calibration.json`
   has an entry for the family, `apply_calibration` rescales the AI probability with Member 3's `(a, b)` and redistributes
   the generator classes proportionally.
5. **Aggregate.** A sentence takes the average of the windows that cover it. A sentence counts as scored only if it has at
   least 20 letters (5 for Chinese). The document AI % is the **length-weighted** average over scored sentences. The
   generator distribution is the AI-weighted average of the window distributions, scaled to the AI share.
6. **Warn.** It adds warnings for notation of 15 % or more, mixed scripts, unsupported scripts (Latin adapter used as the
   fallback) and input shorter than 200 characters (`low_evidence`).
**Safety:** documents beyond a window cap raise an error (default 400 windows, set by `SFACA_MAX_WINDOWS`).
**Output:** a dictionary with `ai_pct`, `human_pct`, `generator_probs`, `top_generator`, `script`, `adapter`,
`routing_confidence`, `mixed_script`, `ndi`, `calibrated`, `low_evidence`, `warnings` and `segments`
(`start`, `end`, `text`, `type`, `ai_prob`, `scored`).
The scorer is injected, so the same code runs against the real model (`InferenceEngine`) and the stub in tests.

### `src/sanitizer.py`
**Objective:** keep math and code out of the score, because equations and code are a common cause of false AI flags.
**Logic:** regular expressions find `$…$`, `$$…$$`, `\(…\)`, `\[…\]`, `equation`, `align`, `matrix` and `gather`
environments, fenced code and inline code. `find_notation_spans(text)` returns `(start, end, type)` spans on the original
text; code wins when the two overlap. The **Notation Density Index** (NDI) is the share of characters inside such spans.
The older `TextSanitizer` and `calibrate_probabilities` (a fixed-factor adjustment) are kept but no longer used, because the
fixed factor was never tested. **Limit:** this is pattern matching. It misses equations with no `$` delimiters and equations
extracted from Word or PDF.

### `src/router.py`
**Objective:** pick the adapter. It counts characters by Unicode block (Latin including accents, Cyrillic, Greek, Arabic
including Persian and Urdu letters, Han, and Japanese kana and Korean Hangul, which are sent to the Chinese family as the
nearest). Hebrew and Devanagari are marked unsupported and fall back to Latin. It returns the winning family, the
routing confidence (its share of letters), and whether several scripts are present.

### `src/file_parser.py`
Reads `.txt`/`.md` (UTF-8 with a Latin-1 fallback), `.pdf` with `pypdf` and `.docx` with `python-docx`, falling back to
unzipping the XML. Unsupported formats raise a clear error.

### `src/exporters.py`
`to_html(result, filename)` builds a self-contained report: the text with each sentence shaded red in three tiers (50 to
65 %, 65 to 80 %, 80 % and above), math and code in dark grey, a legend, the summary and a "likelihood, not proof" note.

### `src/stub_scorer.py`
A deterministic placeholder that makes up plausible scores from a hash and a few word lists. It exists so the API, page and
tests can run without adapters. The API reports `model: "stub"` and the page shows a "Demo mode" banner whenever it is used.

### `backend/api.py`: the server
**Endpoints:** `GET /health`; `POST /api/analyze` (form field `text` or an uploaded `file`); `POST /api/export` (same input,
returns the HTML report). The page itself is served at `/`.
**Safety:** 8 MB and 150,000-character limits; only `.txt .md .docx .pdf` accepted; files are processed in a temporary
folder and deleted; errors are JSON (`400` too short, `413` too large, `415` wrong type, `422` unreadable or scanned PDF,
`503` model not loaded) with no stack traces; one analysis at a time (a lock) so GPU memory stays safe; CORS origins from
`ALLOWED_ORIGINS` (open by default so the Pages site can call it). It starts even without adapters so `/health` can explain
the problem.
**Settings:** `ADAPTER_DIR`, `SFACA_STUB=1`, `ALLOWED_ORIGINS`, `SFACA_MAX_WINDOWS`, `PORT`.

### `backend/app.py`
The first Gradio dashboard. It is kept for reference and still imported by a test, but the deployed interface is the web page.

### `web/index.html`
One file, no build step. It has paste and upload tabs, a donut chart, generator bars with friendly names, a per-sentence
chart (click a bar to jump to the sentence), the shaded text, warnings, a download button, light and dark themes and a
phone layout. **Finding the API:** `config.js` may set a fixed address; otherwise the page reads `api_url.txt` (published
beside it by the launcher) with a cache-busting query, and re-reads it if a request fails. It retries the health check for
up to two minutes and shows "Waking up the analysis server…" instead of an error, because a sleeping or restarting host
takes time.

### `ops/start_public.ps1` and the `.bat`
**Objective:** share the app from a laptop with one double-click. It starts the API (if not already healthy), opens a free
Cloudflare quick tunnel (retrying, because Cloudflare's tunnel-creation call takes 10 to 16 seconds and `cloudflared`
gives up at 15), writes the new address to `web/api_url.txt`, and commits and pushes it so the page picks it up in 1 to 2
minutes.

### `deploy/`
`Dockerfile` (Python 3.11, CPU PyTorch, the backbone baked into the image, runs `backend/api.py`), `SPACE_README.md`
(Hugging Face Space metadata) and `deploy_space.py` (stages the repository layout and uploads it). **Status:** written but
unused, because Hugging Face now charges for Docker Spaces (HTTP 402); the image was also not test-built.

### `.github/workflows/pages.yml`
Publishes `web/` to GitHub Pages whenever it changes on `main`. Pages must be switched to "GitHub Actions" as its source
once (Settings, Pages).

## 4. How a request travels

```
user's browser ─▶ GitHub Pages (web/index.html) ─▶ reads api_url.txt ─▶ Cloudflare tunnel ─▶ laptop: backend/api.py
      ▲                                                                                               │
      └──────────────── JSON result drawn as chart, bars and shaded text ◀── analyze() ◀── InferenceEngine (GPU)
```
A user needs only the link and a browser. The model and API run on the laptop that started the launcher.

## 5. Problems found and how they were solved

| Problem | Solution |
|:-|:-|
| The old API crashed on an undefined variable and had its own sanitizer using a placeholder never seen in training | One shared pipeline; math and code are excluded, never replaced |
| The old page had a temporary tunnel address hard-coded | The address is configuration, read from `api_url.txt` |
| GitHub Pages cannot run a model | Page on Pages, model on a laptop behind a tunnel |
| Hugging Face Docker Spaces need a paid plan (402) | Bundle prepared; laptop tunnel used meanwhile |
| A free SSH tunnel dropped ("no tunnel") | Switched to `cloudflared` |
| `cloudflared` timed out creating a tunnel | Retry loop in the launcher |
| The tunnel address changes at every restart | `api_url.txt` plus the launcher publishes it automatically |
| The launcher aborted on git's normal stderr output | Error handling relaxed around git commands |
| A sleeping host looked broken | Retry and "waking up" message in the page |

## 6. Evidence

- `python -m pytest 04_pipeline_backend_frontend/tests -q`: 11 tests on offsets, notation, empty and notation-only input,
  mixed scripts, window cap, health, text and file upload (txt, docx), bad type, corrupt PDF, export.
- The live page was exercised end to end through the tunnel with the real model; AI-style paragraphs scored 82 % AI and
  human news sentences 11 to 31 % per sentence.
- Speed: about 100 to 150 ms per document on the RTX 4050; about 0.6 s per window on a 2-thread CPU.

## 7. How to run

```bash
SFACA_STUB=1 python 04_pipeline_backend_frontend/backend/api.py          # demo, no model needed
ADAPTER_DIR=results/adapters python 04_pipeline_backend_frontend/backend/api.py   # real model
# open http://localhost:7860
```
To share it, double-click `04_pipeline_backend_frontend/ops/Start SFA-CA Public.bat`.

## 8. Limits

- The public link works only while the laptop is on and the launcher window is open. This is a demo setup.
- Cloudflare quick tunnels have no uptime guarantee; several users are served one after another.
- Highlighted `.docx` and `.pdf` export is not built (HTML only). Scanned PDFs have no OCR.
- The Docker image has never been built.
- Math detection is pattern-based (see `sanitizer.py`).

## 9. Viva questions

1. **Why overlapping windows?** Each sentence is judged twice in context, and a document of any length can be scored.
2. **Why contiguous segments with offsets?** So the page can shade the exact original text, and a test proves they rejoin.
3. **How can a friend use it without installing anything?** The page loads from GitHub; their text goes through the tunnel to the laptop.
4. **Why a stub scorer?** To test and demo the whole product without trained adapters; the page labels it clearly.
5. **What would make it production-grade?** A permanent host (paid Space or VM), CI, highlighted document export and an evidence page.
