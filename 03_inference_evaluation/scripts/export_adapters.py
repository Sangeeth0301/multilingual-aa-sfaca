"""Collect the best trained run per cluster into the layout the API loads.

    python 03_inference_evaluation/scripts/export_adapters.py --runs results/runs --out results/adapters

Picks, per cluster, the run with the highest best_val_macro_f1 (validation only,
never test) and copies adapter_model.bin + classifier_head.pt + config/metrics.
Then run the app with ADAPTER_DIR=results/adapters.
"""
import argparse
import glob
import json
import os
import shutil

CLUSTERS = ["latin", "cyrillic", "greek", "arabic", "hanzi"]
FILES = ["adapter_model.bin", "classifier_head.pt", "config.json", "metrics.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/adapters")
    args = ap.parse_args()
    for cl in CLUSTERS:
        best, best_f1 = None, -1.0
        for m in glob.glob(os.path.join(args.runs, f"{cl}_seed*", "metrics.json")):
            run = os.path.dirname(m)
            if not os.path.exists(os.path.join(run, "adapter_model.bin")):
                continue
            f1 = json.load(open(m)).get("best_val_macro_f1", -1.0)
            if f1 > best_f1:
                best, best_f1 = run, f1
        if best is None:
            print(f"[skip] {cl}: no finished run found")
            continue
        dst = os.path.join(args.out, cl)
        os.makedirs(dst, exist_ok=True)
        for f in FILES:
            if os.path.exists(os.path.join(best, f)):
                shutil.copy2(os.path.join(best, f), dst)
        print(f"[ok]   {cl}: {best}  val macro-F1 {best_f1:.4f}")


if __name__ == "__main__":
    main()
