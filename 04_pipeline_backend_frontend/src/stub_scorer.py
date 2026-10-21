"""Deterministic heuristic scorer used ONLY for demos/tests when no trained adapters exist.
It is not a detector; the API reports model="stub" so the UI can say so."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)
import hashlib
import re

import numpy as np

from data_loader import LABEL_LIST_8CLASS

AI_WORDS = re.compile(r"\b(moreover|furthermore|delve|crucial|pivotal|multifaceted|in conclusion|overall|underpin)\b", re.I)
HUMAN_WORDS = re.compile(r"\b(i|honestly|my|lol|we got|mess|grumbling)\b", re.I)


class StubScorer:
    name = "stub"
    labels = LABEL_LIST_8CLASS

    def predict_probs(self, texts, cluster):
        rows = []
        for t in texts:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            ai = 0.25 + 0.2 * ((h % 100) / 100)
            ai += 0.2 * len(AI_WORDS.findall(t)) - 0.2 * len(HUMAN_WORDS.findall(t))
            ai = float(np.clip(ai, 0.03, 0.97))
            g = 1 + (h >> 8) % (len(self.labels) - 1)
            row = np.full(len(self.labels), ai * 0.4 / (len(self.labels) - 2))
            row[g] = ai * 0.6
            row[0] = 1 - ai
            rows.append(row / row.sum())
        return np.asarray(rows)
