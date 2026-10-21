# Prompt to paste into Lovable

Build a polished, modern single-page web app called **"AI Text Attribution"** that tells people whether a piece of text was written by a human or an AI, which passages are AI, and which AI model likely wrote it.

## Design
- Clean, premium, SaaS-quality look: generous spacing, soft shadows, rounded cards, one accent colour, smooth transitions, Inter font.
- Light and dark mode with a toggle. Fully responsive, works on a phone.
- Hero section: title, one-line subtitle, then a big input card.
- Input card has two tabs: **Paste text** (large textarea, character counter) and **Upload file** (drag-and-drop zone accepting .pdf, .docx, .txt, .md, max 8 MB). Buttons: **Load sample** and **Analyze**.
- Animated loading state while analysing (skeleton cards or progress shimmer).

## Results (shown below the input after analysis)
1. **Verdict card:** animated donut chart "xx% AI / yy% Human", plus chips for script, adapter, and a "likely source" label. Use wording like "likely" and "evidence", never "proof".
2. **"Which AI wrote it?" card:** horizontal bars for every class in `generator_probs`; highlight the top AI in red. Show friendly names (GPT-3.5 Turbo, Llama-2 70B, Mistral 7B, Vicuna 13B, OPT-IML 30B, Eagle 7B, Aya-101, Human).
3. **Document timeline:** a bar chart with one bar per scored sentence showing AI probability; clicking a bar scrolls to and flashes that sentence below. Notation segments are grey bars.
4. **Highlighted document:** the full text, each sentence shaded in red proportional to `ai_prob` (no shading below 0.35); sentences with `type: "notation"` shown in monospace grey and labelled "math/code, excluded"; tooltip with the percentage; a summary "N of M sentences flagged (≥50% AI)".
5. **Warnings:** show each string in `warnings` as a yellow banner; show a "Low evidence" chip when `low_evidence` is true.
6. **Download highlighted report** button.
7. A small, visible responsible-use note: "This is a statistical likelihood, not proof of authorship. Short, edited or paraphrased text is unreliable."
- If `/health` reports `"model": "stub"`, show a banner "Demo mode: placeholder scores".

## API (already built, do not mock it)
Base URL is a config constant `API_URL` at the top of the code (I will set it).

- `GET {API_URL}/health` returns `{"status":"ok","model":"sfa-ca","labels":[...]}`.
- `POST {API_URL}/api/analyze` with multipart/form-data containing EITHER `text` (string) OR `file` (the uploaded file). Returns:
```json
{
  "ai_pct": 62.4, "human_pct": 37.6,
  "generator_probs": {"human": 0.376, "Llama-2-70b-chat-hf": 0.31, "Mistral-7B-Instruct-v0.2": 0.05, "aya-101": 0.02, "gpt-3.5-turbo-0125": 0.1, "opt-iml-max-30b": 0.01, "v5-Eagle-7B-HF": 0.02, "vicuna-13b": 0.1},
  "top_generator": "Llama-2-70b-chat-hf",
  "script": "Latin", "adapter": "latin", "routing_confidence": 0.98, "mixed_script": false,
  "ndi": 0.04, "low_evidence": false, "warnings": [],
  "segments": [
    {"start": 0, "end": 142, "text": "...", "type": "text", "ai_prob": 0.81, "scored": true},
    {"start": 142, "end": 171, "text": "$E=mc^2$", "type": "notation", "ai_prob": null, "scored": false}
  ],
  "model": "sfa-ca", "filename": "essay.pdf"
}
```
  `segments` are contiguous and concatenate to the original text. A text segment with `scored: false` is too short to score; render it unshaded.
- `POST {API_URL}/api/export` takes the same form data and returns an HTML file; trigger it as a download named `sfaca-report.html`.
- Errors come back as `{"detail": "message"}` with status 400, 413, 415, 422 or 503. Show the `detail` text in a friendly inline error, never a stack trace.

Do not add sign-in, history, sidebars or pricing pages.
