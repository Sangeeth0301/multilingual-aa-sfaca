# Member 3: Inference, calibration and evaluation

## 1. Role and objective

**Objective:** take the trained adapters from Member 2 and make them work on real documents, then measure honestly how
well they work.

A trained model is not yet a detector. Training saw whole news articles cut to 128 tokens. Users upload documents of any
length, sometimes mixed human and AI. This folder closes that gap in three ways:

1. **Inference:** load the five adapters and heads once and score text in batches.
2. **Calibration:** turn the model's raw scores into a believable "percent AI" (this fixed a serious over-flagging problem).
3. **Evaluation:** measure accuracy on held-out documents through the real pipeline, and produce the evidence and reports.

## 2. What is in this folder

| File | Lines | Purpose |
|:-|:-:|:-|
| `src/inference.py` | 162 | `InferenceEngine`: loads adapters and heads, scores text |
| `scripts/calibrate.py` | 93 | Fits the per-script calibration and picks the window size |
| `scripts/eval_pipeline.py` | 59 | End-to-end test on held-out documents |
| `scripts/export_adapters.py` | 45 | Chooses the best run per family and prepares the adapter folder |
| `src/evaluate_cross_cluster.py` | 144 | Tests one family's adapter on another family's data (transfer) |
| `src/low_resource_eval.py` | 147 | Zero-shot test on languages that were never trained |
| `src/push_to_hub.py` | 88 | Uploads adapters to the Hugging Face Hub |
| `scripts/shortcut_audit.py` | 140 | Audits data features that could act as shortcuts |
| `scripts/report_stats.py` | 107 | Computes statistics for the data card |
| `scripts/generate_data_card.py` | 218 | Regenerates `01_data_engineering/DATA_CARD.md` from the results |
| `tests/test_inference.py` | 226 | Tests for routing, head isolation, engine parity, loss, sanitizer, parser |

## 3. How the code works, file by file

### `src/inference.py`: the engine
**Objective:** one place that turns text into class probabilities, correctly and quickly.
**Logic:**
- `load_from_dir(adapter_dir)` builds the model once with **all five adapters**. For each family it reads
  `adapter_model.bin` and loads it into that family's named adapter slot with `set_peft_model_state_dict`, and reads
  `classifier_head.pt` into **that family's own head**, stored in a `ModuleDict`. Heads are loaded on the CPU first and
  then moved, so it also works without a GPU.
- `predict(text, cluster)` scores one text. `predict_probs(texts, cluster, batch_size=16)` scores many in batches under
  `no_grad` and FP16 autocast on a GPU, and returns an array of shape `(n, 8)`. If a family has no adapter loaded it falls
  back to the first one loaded instead of failing.
- Switching families calls `set_adapter` and uses that family's head, so the adapter and head always match. The earlier app
  loaded all heads into one shared head and silently used the wrong one for non-Latin text.
- The 8 class names come from `configs/default.yaml` or the default list; the first is always `human`.

### `scripts/calibrate.py`: the fix that mattered most
**The problem:** the model spreads "AI" over seven generators. A text can have P(human) = 0.30 and still be classed human
because each generator gets only about 0.10. Taking `1 − P(human)` as "percent AI" then reports 70 % AI for text the model
would call human. On held-out Cyrillic human text the first full test gave **93 % AI on average**.
**Logic:**
1. Take validation documents only. Split each into overlapping windows with the same code the app uses.
2. **Choose the window size** among 64, 100 and 150 tokens by validation AUROC: 0.943, 0.922 and 0.908. 64 wins.
3. For each family compute `p_ai = 1 − P(human)` for every window and fit **Platt scaling**:
   `calibrated = sigmoid(a·logit(p_ai) + b)` with a class-balanced logistic regression. The fitted `b` is strongly negative
   (about −2.4 to −3.8), which is exactly the correction for over-estimating AI.
4. Write `configs/calibration.json` (per-family `a`, `b` and the window size). Member 4's analyzer reads it and, after
   scaling, redistributes the generator probabilities in proportion so each row still sums to 1.
**Rule kept:** only validation data is used; the test split is never touched here.

### `scripts/eval_pipeline.py`: honest end-to-end evaluation
**Objective:** measure what a user would experience, not just classifier metrics.
**Logic:** (1) for each family, take held-out test documents longer than 400 characters (up to 25 human and 25 AI), run each
through the full pipeline and record AI %, the top generator and the time; accuracy counts AI % ≥ 50 for AI text and
< 50 for human text. (2) **Mixed-document test:** build 10 English documents from 3 human then 3 GPT-3.5 paragraphs, score
every sentence and compare with the known truth to get precision, recall and F1 for the AI class.

### `scripts/export_adapters.py`
Looks in `results/runs/<cluster>_seed*/`, picks the run with the highest **validation** macro-F1 for each family, and copies
`adapter_model.bin`, `classifier_head.pt`, `config.json` and `metrics.json` to `results/adapters/<cluster>/`, the layout the
API loads.

### `src/evaluate_cross_cluster.py`, `src/low_resource_eval.py`, `src/push_to_hub.py`
- `evaluate_cross_cluster` loads one family's adapter and head and evaluates it on any family's data, producing the
  train-family versus test-family transfer matrix.
- `low_resource_eval` tests languages absent from training (for example Catalan, Irish, Scottish Gaelic) with the nearest
  family's adapter.
- `push_to_hub` uploads each family's adapter folder to a Hugging Face repository.
**Status:** these three were written before the retraining and were **not re-run after it**; whether they match the adapter
layout that `train.py` now writes has not been checked (see section 7).

### `scripts/shortcut_audit.py`
**Objective:** find features that could let a model cheat. For each family and generator it reports character and token
length, the share of texts beyond 128 tokens, script-mismatch rate, boilerplate rate, the most common opening words and
text-hash overlaps. Finding: generators differ strongly in length (for Cyrillic, OPT-IML averages 266 characters and Aya
942), and in Cyrillic 59 to 89 % of the texts (75 % of the human ones) are longer than the 128-token window, which means
training saw only the opening of those texts. Report: `results/shortcut_audit.md`.

### `scripts/report_stats.py` and `scripts/generate_data_card.py`
`report_stats` joins the splits with the raw file's `source` column and computes group counts and source/outlet statistics
into `results/data_card_metrics.*`.
`generate_data_card` writes `01_data_engineering/DATA_CARD.md` from those results. Running it **overwrites** the data card.

### `tests/test_inference.py`
Tests: a 15-case router table (Latin, Cyrillic, Greek, Arabic, Persian, Urdu, Chinese, Japanese, Korean, mixed text,
unsupported scripts); **head isolation** (two families give different heads); **engine parity** (the engine's logits equal
the model's own); a missing `ADAPTER_DIR` setting raises a clear error; the contrastive loss never returns NaN; the sanitizer; and
the parser. It also covers modules owned by other members.

## 4. Results

Full-pipeline test (`results/pipeline_eval.md`, produced by `eval_pipeline.py`; test split; single seed):

| Family | Human docs | AI docs | Human-vs-AI accuracy | Generator top-1 | Latency p50 |
|:-|:-:|:-:|:-:|:-:|:-:|
| Latin | 25 | 25 | 0.98 | 0.72 | 127 ms |
| Cyrillic | 25 | 25 | 0.82 | 0.36 | 126 ms |
| Greek | 25 | 25 | 0.90 | 0.84 | 100 ms |
| Arabic | 25 | 25 | 0.98 | 0.84 | 94 ms |
| Chinese | 4 | 25 | 0.83 | 0.48 | 124 ms |

Mixed English documents, sentence level: precision 0.827, recall 1.000, F1 0.906 (259 true positives, 54 false positives,
0 false negatives, 295 true negatives).

Effect of calibration: before calibration Latin accuracy was 0.94 and Cyrillic 0.52 (human text averaged 93 % AI); after
calibration they are 0.98 and 0.82 (human text 33 % AI).

## 5. Interfaces

| | Item |
|:-|:-|
| **Receives** | `results/runs/*` from Member 2; `load_training_table` from Member 1 |
| **Provides** | `InferenceEngine` (`labels`, `predict_probs`), `configs/calibration.json`, `results/adapters/` |
| **Provides to Member 4** | The scorer interface the analyzer calls |

## 6. How to run and verify

```bash
python 03_inference_evaluation/scripts/export_adapters.py          # runs → results/adapters
python 03_inference_evaluation/scripts/calibrate.py                # writes configs/calibration.json (validation only)
python 03_inference_evaluation/scripts/eval_pipeline.py            # prints the table above
python -m pytest 03_inference_evaluation/tests -q
```
Re-run `calibrate.py` every time the adapters change, because the calibration belongs to one set of adapters.

## 7. Limits

- Samples are small (25 per class) and the seed is single, so numbers are indicative. Chinese has only 4 human test documents.
- Highlighting was measured only in English; Cyrillic generator naming (0.36) is weak.
- Calibration is fitted on validation windows from news text; other domains may shift it.
- `evaluate_cross_cluster`, `low_resource_eval` and `push_to_hub` need a compatibility check against the current adapter layout.
- Windows of 64 tokens are shorter than the 128-token training texts; short windows are noisier.

## 8. Viva questions

1. **Why can "percent AI" be wrong even when the classifier is accurate?** The AI probability is split over seven generators, so `1 − P(human)` over-states AI; calibration fixes the scale.
2. **Why Platt scaling, and why validation only?** It is a simple, well-understood monotone mapping with two numbers per family; using test data to fit it would inflate the result.
3. **Why windows?** The model was trained on short fixed-length texts; windows let a long document be scored in pieces, which also gives per-sentence localisation.
4. **What does engine parity prove?** That the production path produces exactly the same logits as the training model.
5. **What would you do to improve Cyrillic?** More data and seeds, check the shortcut audit's length differences, and re-fit the calibration.
