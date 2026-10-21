"""Fit per-cluster Platt calibration of P(AI) on VALIDATION documents, using the same
window pipeline the app uses. Writes configs/calibration.json.

    python 03_inference_evaluation/scripts/calibrate.py --adapters results/adapters

Window size is chosen by validation AUROC (never test).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import sfaca_paths  # noqa: E402,F401
os.chdir(ROOT)

from analyzer import build_windows, segment  # noqa: E402


def doc_windows(text, window_tokens):
    segs = segment(text)
    idx = [i for i, s in enumerate(segs) if s["type"] == "text" and s["text"].strip()]
    return [" ".join(segs[k]["text"].strip() for k in w) for w in build_windows(idx, segs, window_tokens)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="results/adapters")
    ap.add_argument("--per_class", type=int, default=120)
    ap.add_argument("--windows", default="64,100,150")
    args = ap.parse_args()
    os.environ["ADAPTER_DIR"] = os.path.abspath(args.adapters)
    from data_loader import load_training_table
    from inference import InferenceEngine

    eng = InferenceEngine()
    eng.load_from_dir()
    df, _ = load_training_table()
    val = df[(df["split"] == "val") & (df["text"].str.len() > 300)]
    sizes = [int(x) for x in args.windows.split(",")]

    # 1) choose window size by pooled val AUROC on uncalibrated P(AI)
    cache, best_w, best_auc = {}, None, -1
    for w in sizes:
        ys, ps = [], []
        for cl in eng.classifiers:
            sub = val[val["cluster"] == cl]
            hum = sub[sub["multi_label"] == "human"].sample(min(60, (sub["multi_label"] == "human").sum()), random_state=0)
            ai = sub[sub["multi_label"] != "human"].sample(min(60, (sub["multi_label"] != "human").sum()), random_state=0)
            for lab, part in ((0, hum), (1, ai)):
                for t in part["text"]:
                    ws = doc_windows(t, w)
                    if ws:
                        ys.append(lab)
                        ps.append(float(np.mean(1 - eng.predict_probs(ws, cl)[:, 0])))
        auc = roc_auc_score(ys, ps)
        print(f"window={w}: val AUROC {auc:.4f}")
        if auc > best_auc:
            best_w, best_auc = w, auc

    # 2) fit Platt scaling per cluster at the chosen window size, on window-level P(AI)
    cal = {"_window_tokens": best_w}
    for cl in eng.classifiers:
        sub = val[val["cluster"] == cl]
        X, y = [], []
        for lab, part in ((0, sub[sub["multi_label"] == "human"]), (1, sub[sub["multi_label"] != "human"])):
            part = part.sample(min(args.per_class, len(part)), random_state=1)
            for t in part["text"]:
                ws = doc_windows(t, best_w)
                if ws:
                    p_ai = 1 - eng.predict_probs(ws, cl)[:, 0]
                    X += list(logit(p_ai))
                    y += [lab] * len(ws)
        lr = LogisticRegression(C=1e4, class_weight="balanced").fit(np.array(X).reshape(-1, 1), y)
        cal[cl] = {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0]), "n_windows": len(y)}
        print(cl, cal[cl])
    out = os.path.join(ROOT, "configs", "calibration.json")
    json.dump(cal, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
