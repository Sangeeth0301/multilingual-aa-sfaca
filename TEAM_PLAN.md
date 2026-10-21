# Team plan: four members, one system

SFA-CA detects AI-written text in five script families, finds which sentences are AI-written, and names the most
likely generator. The work is split into **four folders, one per member**. Each folder holds that member's code,
tests and a `REPORT.md` that explains the objective and the logic of every file.

```
README.md              project overview (shared)
TEAM_PLAN.md           this file
configs/               shared configuration (default.yaml, calibration.json)
requirements*.txt      shared dependencies
sfaca_paths.py         one helper that lets the four folders import each other
data/  results/        shared inputs and outputs (not source code)

01_data_engineering/           Member 1  builds trustworthy data
02_model_training/             Member 2  builds and trains the model
03_inference_evaluation/       Member 3  runs, calibrates and measures the model
04_pipeline_backend_frontend/  Member 4  turns the model into a product and deploys it
```

## 1. Who owns what

| | Member 1 | Member 2 | Member 3 | Member 4 |
|:-|:-|:-|:-|:-|
| **Role** | Data engineer | ML engineer | Inference and evaluation engineer | Product, backend and deployment engineer |
| **One-line mission** | Make the data clean, leak-free and reproducible | Train one adapter per script family | Make the trained model correct and measurable on real documents | Make it usable by anyone with a link |
| **Folder** | `01_data_engineering` | `02_model_training` | `03_inference_evaluation` | `04_pipeline_backend_frontend` |
| **Files in the folder** | 11 | 13 | 11 | 21 |
| **Lines of code** | 1,609 | 1,057 | 1,203 | 1,263 |
| **Lines of tests** | 105 | 454 | 225 | 98 |
| **Code + tests** | **1,714** | **1,511** | **1,428** | **1,361** |
| **Share of the code** | 28.5 % | 25.1 % | 23.7 % | 22.6 % |

Lines are only part of the effort, so each member also owns work that is not counted in lines. The totals are within
about 14 % of the average once that work is included.

| Member | Work that is not measured in lines |
|:-|:-|
| 1 | Data audit and the decision to drop the fake Gemini/Claude rows, the frozen split hash, the data card, dataset reproducibility |
| 2 | About 2 hours of GPU training for 5 clusters, hyperparameter and seed decisions, all training-time debugging, the Kaggle notebooks |
| 3 | The calibration study that fixed the "human text flagged as AI" problem, the full-pipeline evaluation, adapter export and Hugging Face publishing |
| 4 | UI design, API contract, tunnel and launcher, GitHub Pages and the Docker bundle, the pull request and the project documentation |

## 2. How the four parts connect (the contracts)

```
 01 data  ──splits_8class.csv + data_loader──▶  02 training ──adapter_model.bin + classifier_head.pt──▶  03 inference
                                                                                                           │
 03 calibration.json + InferenceEngine.predict_probs()  ◀──────────────────────────────────────────────────┘
                     │
                     ▼
 04 analyzer.analyze(text, scorer) ──JSON──▶ FastAPI /api/analyze ──▶ web page (GitHub Pages) ──▶ the user
```

| Hand-off | From → to | The contract (what must not change without telling the other member) |
|:-|:-|:-|
| Split file | 1 → 2 | `data/processed/splits_8class.csv`, frozen SHA-256 `8c05df83…`, columns `row_id, cluster, split, group_id` |
| Loader API | 1 → 2, 3 | `load_training_table()`, `LABEL_LIST_8CLASS` (order: human, Llama-2, Mistral, Aya, GPT-3.5, OPT-IML, Eagle, Vicuna), `get_cluster()` |
| Adapter files | 2 → 3 | `results/runs/<cluster>_seed<k>/{adapter_model.bin, classifier_head.pt, metrics.json}` |
| Exported adapters | 3 → 4 | `results/adapters/<cluster>/…` and `configs/calibration.json` |
| Scorer interface | 3 → 4 | `scorer.labels` and `scorer.predict_probs(texts, cluster) -> array[n, 8]` |
| Response JSON | 4 → web | fields listed in `04_pipeline_backend_frontend/REPORT.md` |

Rule: **a contract change needs a short message to the receiving member and an updated test**, because the pieces are
developed in parallel.

## 3. Schedule (4 weeks, all four working in parallel)

| Week | Member 1 | Member 2 | Member 3 | Member 4 |
|:-|:-|:-|:-|:-|
| **1. Foundations** | Audit the raw data, define the clusters, write `inspect_data` | Build `model.py` and the contrastive loss with unit tests | Define the scorer interface, write the stub scorer with Member 4 | Define the API and response JSON, build the page against the stub scorer |
| **2. Build** | `build_splits`, flags, similarity, `check_leakage`, freeze the split hash | `train.py` (validation-only selection, resume, AMP), smoke test, shuffled-label sanity run | `InferenceEngine`, batched scoring, `export_adapters` | `analyzer` (sanitize, segment, windows, aggregate), file parser, FastAPI |
| **3. Train and measure** | Data card, reproducibility check, hand over final splits | Train all 5 clusters, then seeds 43 and 44 | Calibration, full-pipeline evaluation, localization test | Highlighted report export, deploy bundle, tunnel and launcher |
| **4. Integrate and present** | Review and finalize data docs | Review and finalize training docs | Final numbers into the README and reports | End-to-end test on the live page, pull request, final demo |

Joint checkpoints: end of week 1 (contracts agreed), end of week 2 (first end-to-end run with the stub),
end of week 3 (real model in the page), week 4 (rehearsal).

## 4. Rules that keep the work equal and the code correct

1. **Every member reviews one other member's pull request**: 1 reviews 2, 2 reviews 3, 3 reviews 4, 4 reviews 1.
2. **Each member explains their own part in the viva**: five minutes each (section 6).
3. **No number goes into a document unless a script in the repository produced it.** The script name goes next to it.
4. **The test set is touched once**, at the end of training. Choices (epoch, window size, calibration) use validation only.
5. **`pytest -q` from the repository root must pass** before anything is merged (currently 47 tests).
6. **Commit messages** name the member folder, for example `02: fix resume parity`.

## 5. Next work, split equally (three to four tasks each)

| Member | Task | Done when |
|:-|:-|:-|
| 1 | Add real text from newer generators (GPT-4o, Claude, Gemini) with a story-grouped split, after checking licences | `check_leakage.py` passes; data card updated |
| 1 | Run `check_leakage.py` automatically in CI | A failing leak blocks a merge |
| 1 | Add Indic and African low-resource test data (the earlier Tamil run had none) | Real zero-shot numbers, or an honest "not evaluated" |
| 2 | Train seeds 43 and 44 for all five clusters | Mean and standard deviation per cluster |
| 2 | Train the single-adapter baseline (`--cluster all`) | A measured SFA-CA versus baseline comparison |
| 2 | Ablation: contrastive weight 0 versus 0.5; LoRA rank 8, 16, 32 | A table in `02_model_training/REPORT.md` |
| 3 | Re-fit calibration on the seed ensemble | New `configs/calibration.json`, evaluation re-run |
| 3 | Measure highlighting for Cyrillic, Greek, Arabic and Chinese | A table like the English one |
| 3 | Build a labelled math/code benchmark (20 human + 20 AI documents) | False-positive rate with and without the sanitizer |
| 4 | Move to a permanent host (paid Hugging Face Space or a small VM) | The link works with every laptop off |
| 4 | Export highlighted `.docx` and `.pdf` | Output opens in Word and Acrobat |
| 4 | Add pytest to GitHub Actions and an evidence tab to the page | Green CI badge; the tab reads real metrics |

## 6. Presentation: four equal slots

| Slot | Speaker | Content (about 5 minutes) |
|:-|:-|:-|
| 1 | Member 1 | The problem, the data, why the splits are leak-free, what we removed and why |
| 2 | Member 2 | The model idea (script-family adapters, contrastive loss), training setup, results per family |
| 3 | Member 3 | How the model is used on real documents, the calibration fix, the evaluation numbers, limits |
| 4 | Member 4 | Live demo (paste, upload, highlight), architecture of the backend, how it is deployed |

## 7. Risks and who watches them

| Risk | Owner | Mitigation |
|:-|:-|:-|
| Results change between training runs | 2 | Fixed seeds, three seeds, report mean and standard deviation |
| Newer AI models are not recognised | 1 and 3 | Add newer generators to the data (task 1), report the limit in the UI |
| Free tunnel or laptop goes offline | 4 | Launcher script, wake-up messages in the page, permanent host task |
| A contract changes silently | all | Contract table above plus tests |
| Over-claiming accuracy | 3 | Rule 3: every number comes from a script |
