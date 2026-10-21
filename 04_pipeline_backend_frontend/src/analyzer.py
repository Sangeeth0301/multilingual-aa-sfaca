"""
analyzer.py
-----------
Document-level analysis pipeline: sanitize -> segment -> window -> route ->
score -> aggregate. The scorer is injected, so the same code path runs against
the real SFA-CA engine and against a stub in tests.

A scorer must provide:
    labels: list[str]                      class names, labels[0] == "human"
    predict_probs(texts, cluster) -> array of shape (len(texts), len(labels))
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import os
import re
from typing import Dict, List, Optional

import numpy as np

from router import detect_script
from sanitizer import find_notation_spans

SENT_RE = re.compile(r"[^.!?。！？؟\n]+(?:[.!?。！？؟]+|\n+|$)\s*")
CJK_RE = re.compile(r"[぀-ヿ一-鿿가-힯]")

MIN_SENTENCE_LETTERS = 20      # below this a sentence is too short to carry signal
MIN_DOC_CHARS = 200            # below this the whole document is low evidence
MAX_WINDOWS = int(os.environ.get("SFACA_MAX_WINDOWS", 400))   # hard cap per request (lower on CPU hosts)
NDI_WARN = 0.15


def est_tokens(text: str) -> float:
    """Cheap token estimate (no tokenizer needed): ~4 chars/token, ~1.3 for CJK."""
    cjk = len(CJK_RE.findall(text))
    return cjk / 1.3 + (len(text) - cjk) / 4.0


def _split_sentences(text: str, base: int) -> List[Dict]:
    out = []
    for m in SENT_RE.finditer(text):
        if m.end() > m.start():
            out.append({"start": base + m.start(), "end": base + m.end()})
    return out


def segment(text: str) -> List[Dict]:
    """Contiguous segments covering `text` exactly: notation spans + sentences."""
    segs, pos = [], 0
    for s, e, kind in find_notation_spans(text):
        if s > pos:
            segs += [dict(x, type="text") for x in _split_sentences(text[pos:s], pos)]
        segs.append({"start": s, "end": e, "type": "notation", "kind": kind})
        pos = e
    if pos < len(text):
        segs += [dict(x, type="text") for x in _split_sentences(text[pos:], pos)]
    for sg in segs:
        sg["text"] = text[sg["start"]:sg["end"]]
    return segs


def build_windows(sentence_idx: List[int], segs: List[Dict], window_tokens: int):
    """Overlapping windows (stride ~ half) over runs of consecutive text sentences."""
    windows, i, n = [], 0, len(sentence_idx)
    while i < n:
        j, tok = i, 0.0
        while j < n and (tok < window_tokens or j == i):
            tok += est_tokens(segs[sentence_idx[j]]["text"])
            j += 1
        windows.append(sentence_idx[i:j])
        if j >= n:
            break
        half = max(1, (j - i) // 2)
        i += half
    return windows


def apply_calibration(p: np.ndarray, ab) -> np.ndarray:
    """Platt-scale P(AI) = 1 - P(human) with (a, b) fitted on validation windows, then
    rescale the generator classes proportionally so each row still sums to 1."""
    a, b = ab
    ai = np.clip(1.0 - p[:, 0], 1e-6, 1 - 1e-6)
    new_ai = 1.0 / (1.0 + np.exp(-(a * np.log(ai / (1 - ai)) + b)))
    out = p.copy()
    out[:, 0] = 1.0 - new_ai
    rest = p[:, 1:].sum(axis=1, keepdims=True)
    out[:, 1:] = p[:, 1:] / np.clip(rest, 1e-9, None) * new_ai[:, None]
    return out


def load_calibration(path=None):
    import json, os
    path = path or os.path.join(os.path.dirname(__file__), "..", "..", "configs", "calibration.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def analyze(text: str, scorer, window_tokens: int = None, temperature: float = 1.0, calibration: dict = None) -> Dict:
    cal = load_calibration() if calibration is None else calibration
    if window_tokens is None:
        window_tokens = cal.get("_window_tokens", 100)
    text = text.replace("\r\n", "\n")
    segs = segment(text)
    text_idx = [i for i, s in enumerate(segs) if s["type"] == "text" and s["text"].strip()]
    notation_chars = sum(s["end"] - s["start"] for s in segs if s["type"] == "notation")
    ndi = notation_chars / max(1, len(text))
    doc_route = detect_script(text)
    labels = list(scorer.labels)
    n_cls = len(labels)

    warnings: List[str] = []
    result = {
        "ai_pct": 0.0, "human_pct": 100.0,
        "generator_probs": {l: (1.0 if l == "human" else 0.0) for l in labels},
        "top_generator": None,
        "script": doc_route["cluster"].capitalize(), "adapter": doc_route["cluster"],
        "routing_confidence": doc_route["routing_confidence"],
        "mixed_script": doc_route["mixed"],
        "ndi": round(ndi, 4), "calibrated": False, "low_evidence": False,
        "warnings": warnings, "n_windows": 0,
    }
    for s in segs:
        s["ai_prob"] = None
        s["scored"] = False
    result["segments"] = segs

    if not text_idx:
        result["low_evidence"] = True
        warnings.append("No scorable prose found (empty, or only math/code).")
        return result

    windows = build_windows(text_idx, segs, window_tokens)
    if len(windows) > MAX_WINDOWS:
        raise ValueError(f"Document too long: {len(windows)} windows (max {MAX_WINDOWS}).")

    texts = [" ".join(segs[k]["text"].strip() for k in w) for w in windows]
    clusters = [detect_script(t)["cluster"] for t in texts]
    probs = np.zeros((len(windows), n_cls))
    for cl in set(clusters):
        idx = [i for i, c in enumerate(clusters) if c == cl]
        p = np.asarray(scorer.predict_probs([texts[i] for i in idx], cl), dtype=float)
        if temperature != 1.0:
            p = np.power(np.clip(p, 1e-9, 1), 1.0 / temperature)
            p /= p.sum(axis=1, keepdims=True)
        if cl in cal:
            p = apply_calibration(p, (cal[cl]["a"], cal[cl]["b"]))
        probs[idx] = p
    result["calibrated"] = any(c in cal for c in set(clusters))
    result["n_windows"] = len(windows)
    if len(set(clusters)) > 1:
        result["mixed_script"] = True

    # per-sentence class distribution = mean over covering windows
    acc = {k: [] for k in text_idx}
    for w, p in zip(windows, probs):
        for k in w:
            acc[k].append(p)
    sent_prob = {k: np.mean(v, axis=0) for k, v in acc.items()}

    weights, ai_vals, gen_rows = [], [], []
    for k, dist in sent_prob.items():
        ai = float(1.0 - dist[0])
        sg = segs[k]
        letters = sum(c.isalpha() for c in sg["text"])
        sg["ai_prob"] = round(ai, 4)
        sg["scored"] = letters >= MIN_SENTENCE_LETTERS or (CJK_RE.search(sg["text"]) is not None and letters >= 5)
        if not sg["scored"]:
            sg["too_short"] = True
            continue
        weights.append(len(sg["text"]))
        ai_vals.append(ai)
        gen_rows.append(dist)

    if not weights:
        result["low_evidence"] = True
        warnings.append("Every sentence is too short to score reliably.")
        return result

    w = np.asarray(weights, float)
    ai_pct = float(np.average(ai_vals, weights=w))
    dists = np.asarray(gen_rows)
    gen_ai = (dists[:, 1:] * (w * 1.0)[:, None]).sum(axis=0)
    gen_ai = gen_ai / max(gen_ai.sum(), 1e-9) * ai_pct
    gp = {"human": 1.0 - ai_pct}
    for lbl, v in zip(labels[1:], gen_ai):
        gp[lbl] = float(v)
    result.update({
        "ai_pct": round(ai_pct * 100, 2), "human_pct": round((1 - ai_pct) * 100, 2),
        "generator_probs": {k: round(v, 4) for k, v in gp.items()},
        "top_generator": max(labels[1:], key=lambda l: gp[l]),
    })

    if len(text.strip()) < MIN_DOC_CHARS:
        result["low_evidence"] = True
        warnings.append("Very short input: the score is unreliable.")
    if ndi >= NDI_WARN:
        warnings.append(f"{ndi*100:.0f}% of the text is math/code and was excluded from scoring.")
    if result["mixed_script"]:
        warnings.append("Several scripts detected: each passage was routed to its own adapter.")
    if not doc_route["supported"]:
        warnings.append("Script not covered by any adapter; Latin adapter used as the nearest fallback.")
    return result
