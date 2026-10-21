import os, sys, time, json
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import sfaca_paths  # noqa: E402,F401
os.chdir(ROOT)
os.environ["ADAPTER_DIR"] = os.path.join(ROOT, "results", "adapters")
from inference import InferenceEngine
from analyzer import analyze
from data_loader import load_training_table

t0 = time.time()
eng = InferenceEngine(); eng.load_from_dir()
print("load s", round(time.time() - t0, 1), "heads", list(eng.classifiers.keys()), "device", eng.device)

df, _ = load_training_table()
test = df[df["split"] == "test"]
rng = np.random.RandomState(0)
out = {}
for cl in ["latin", "cyrillic", "greek", "arabic", "hanzi"]:
    sub = test[test["cluster"] == cl]
    sub = sub[sub["text"].str.len() > 400]
    hum = sub[sub["multi_label"] == "human"]; hum = hum.sample(min(25, len(hum)), random_state=1)
    ai = sub[sub["multi_label"] != "human"]; ai = ai.sample(min(25, len(ai)), random_state=1)
    res = {"human_ai_pct": [], "ai_ai_pct": [], "gen_hits": 0, "ms": []}
    for _, r in hum.iterrows():
        t = time.time(); a = analyze(r["text"], eng); res["ms"].append((time.time() - t) * 1000)
        res["human_ai_pct"].append(a["ai_pct"])
    for _, r in ai.iterrows():
        t = time.time(); a = analyze(r["text"], eng); res["ms"].append((time.time() - t) * 1000)
        res["ai_ai_pct"].append(a["ai_pct"])
        res["gen_hits"] += int(a["top_generator"] == r["multi_label"])
    h = np.array(res["human_ai_pct"]); x = np.array(res["ai_ai_pct"])
    acc = ((h < 50).sum() + (x >= 50).sum()) / (len(h) + len(x))
    out[cl] = dict(binary_acc=acc, human_mean_ai=h.mean().round(1), ai_mean_ai=x.mean().round(1),
                   n=(len(h),len(x)), gen_top1=round(res["gen_hits"] / len(x), 2), ms_p50=int(np.median(res["ms"])), ms_p95=int(np.percentile(res["ms"], 95)))
    print(cl, out[cl])

# mixed document: 3 human paragraphs + 3 AI paragraphs (latin), measure sentence-level localisation
sub = test[(test["cluster"] == "latin") & (test["language"] == "en") & (test["text"].str.len() > 500)]
hum = sub[sub["multi_label"] == "human"].sample(30, random_state=2)["text"].tolist()
gen = sub[sub["multi_label"] == "gpt-3.5-turbo-0125"].sample(30, random_state=2)["text"].tolist()
tp = fp = fn = tn = 0
for i in range(10):
    hs, as_ = hum[i * 3:(i + 1) * 3], gen[i * 3:(i + 1) * 3]
    parts = [(h, 0) for h in hs] + [(a, 1) for a in as_]
    doc, pos, lab = "", [], []
    for p, l in parts:
        pos.append((len(doc), len(doc) + len(p), l)); doc += p + "\n\n"
    a = analyze(doc, eng)
    for s in a["segments"]:
        if not s["scored"]:
            continue
        mid = (s["start"] + s["end"]) // 2
        truth = next(l for st, en, l in pos if st <= mid < en + 2)
        pred = s["ai_prob"] >= 0.5
        tp += pred and truth; fp += pred and not truth; fn += (not pred) and truth; tn += (not pred) and not truth
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
print("localisation(sentence-level, 10 mixed docs): precision", round(prec, 3), "recall", round(rec, 3), "f1", round(2 * prec * rec / max(prec + rec, 1e-9), 3), dict(tp=tp, fp=fp, fn=fn, tn=tn))
