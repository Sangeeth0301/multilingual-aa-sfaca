# Member 2: Model and training

## 1. Role and objective

**Objective:** build the neural model and train it so it can say, for a piece of text, whether it is human or which of
seven AI generators wrote it.

The idea (SFA-CA, Script-Family-Aware Contrastive Adaptation): keep one large multilingual network frozen, and teach it each
writing system by training only a tiny add-on (an adapter) per script family. Five families give five small adapters
instead of one model per language, and the adapters are cheap to train and store (about 7 MB each).

## 2. What is in this folder

| File | Lines | Purpose |
|:-|:-:|:-|
| `src/model.py` | 110 | The model: frozen backbone, one LoRA adapter per family, classifier head |
| `src/contrastive_loss.py` | 105 | The supervised contrastive loss and the combined loss |
| `src/compat.py` | 22 | A small fix so the PEFT library imports on machines with an old `torchao` |
| `src/train.py` | 688 | The full training pipeline |
| `scripts/train_all_local.sh` | 8 | Trains the five families one after another |
| `scripts/shortcut_baselines.py` | 124 | Cheap baselines that test for shortcuts |
| `tests/test_train.py` | 454 | Ten training tests that run on a CPU |
| `notebooks/kaggle/` | | The Kaggle notebooks used for the first runs |
| `EXPERIMENTS.md` | 172 | The planned experiments in priority order, with commands |
| `PHASE_C_REPORT.md` | 100 | The earlier training-phase report |

## 3. How the code works, file by file

### `src/model.py`
**Objective:** one network, five swappable specialisations.
**Logic:**
- Loads **XLM-RoBERTa-large** (about 560 million parameters, trained on 100 languages) and **freezes every weight**.
- Wraps it with Hugging Face PEFT and attaches **LoRA** adapters named `latin`, `cyrillic`, `greek`, `arabic`, `hanzi`.
  LoRA adds two thin matrices (rank 8, scaling 16, dropout 0.05) next to the attention `query` and `value` layers. Only
  these thin matrices train: about **1.84 million** parameters, 0.3 % of the model.
- Adds a **classifier head**: `Linear(h,h) → GELU → Dropout → Linear(h,8)` that maps the first-token ("CLS") vector to 8
  class scores (human plus seven generators).
- `set_active_cluster(name)` switches which adapter is used, and `forward(...)` runs backbone and head. `get_embedding(...)`
  returns the CLS vector for the contrastive loss.

### `src/contrastive_loss.py`
**Objective:** teach the model that texts from the same generator should look alike, and from different generators should
not.
**Logic:** a supervised contrastive loss (SupCon). It normalizes each text's vector, computes all pairwise cosine
similarities divided by a temperature (0.1), then for each text pulls closer the other texts with the same label and
pushes apart the rest. Details that matter: the self-similarity is masked out; a row maximum is subtracted for numerical
stability; and `torch.where` is used instead of multiplying by `-inf` (which would give NaN); texts with no positive
partner in the batch contribute zero. `combined_loss = (1 − w)·cross-entropy + w·contrastive`, with `w = 0.5` by default.
In one-family training the loss works within a family; only the baseline mode `--cluster all` mixes scripts in a batch.

### `src/compat.py`
**Objective:** avoid a crash on Kaggle. PEFT raises an error when `torchao` is older than 0.16; this file makes PEFT treat
`torchao` as unavailable instead.

### `src/train.py`: the training pipeline
**Objective:** train one family (or the all-in-one baseline) reproducibly and report honest numbers.
**Steps in order:**
1. **Skip if finished:** a `DONE.json` file in the run folder means the run is complete.
2. **Load data** with Member 1's `load_training_table`, which verifies the splits hash.
3. **Optional modes:** `--smoke` (1 epoch, 500 rows, prints seconds per 1,000 rows), `--shuffle_labels` (shuffles training
   labels only, a sanity check that results fall to chance), `--max_train_samples` (seeded, class-balanced cap).
4. **Seed everything** (`set_seeds`) so the run can be repeated.
5. **Build model and loaders;** `AdamW` on the trainable parameters only; optional linear schedule; mixed precision (FP16
   autocast with a gradient scaler); optional gradient clipping.
6. **Resume:** after every epoch it saves adapter, head, optimizer, scaler and a `resume_state.json`, so an interrupted run
   continues with identical results.
7. **Epoch loop:** compute the combined loss, back-propagate, evaluate on **validation**, and keep a checkpoint only when
   validation macro-F1 improves. **The test split is not used for any choice.**
8. **Final test, once:** load the best checkpoint and evaluate on test one time.
9. **Write the evidence:** `metrics.json` (macro-F1, accuracy, per-class F1, confusion matrix, human-versus-AI F1 and AUROC,
   scores on the story-disjoint and flag-clean subsets), `config.json` (every argument, the git commit, package versions,
   data hashes) and `DONE.json`.
**Outputs per run folder** `results/runs/<cluster>_seed<k>/`: `adapter_model.bin`, `classifier_head.pt`, `val_probs.npy`,
`metrics.json`, `config.json`.

### `scripts/shortcut_baselines.py`
**Objective:** check the task is not solvable by cheap tricks. It trains simple classifiers on length features, the first
three words, and the quality flags. They score a macro-F1 of 0.05 to 0.43 (chance is 0.125), so length and opening words
carry some signal, but far less than the trained model (0.60 to 0.86). Report: `results/shortcut_baselines.md`.

### `scripts/train_all_local.sh`
Runs `train.py` for Greek, Arabic, Chinese, Cyrillic and Latin with batch size 16, 3 epochs, length 128, learning rate 2e-4,
seed 42, logging to `results/train_<cluster>.log`.

### `tests/test_train.py`
Ten CPU tests using a tiny mock backbone: LoRA attaches to the right layers; the subsample is deterministic and balanced;
shuffling touches only training labels; a wrong splits hash aborts; the same seed gives the same loss curve; the best epoch
is chosen on validation, not test; test data is read exactly once; `--cluster all` merges families; a `DONE.json` skips a
run; resuming after one epoch matches an uninterrupted run.

## 4. Settings and results

| Setting | Value |
|:-|:-|
| Backbone | `xlm-roberta-large`, frozen |
| LoRA | rank 8, alpha 16, dropout 0.05, on `query` and `value` |
| Training | 3 epochs, batch 16, length 128, lr 2e-4, FP16, seed 42, contrastive weight 0.5, temperature 0.1 |
| Hardware | NVIDIA RTX 4050 Laptop GPU (6 GB), about 57 samples per second |

| Family | Train time | Validation F1 | Test macro-F1 | Test accuracy | Human-vs-AI AUROC |
|:-|:-:|:-:|:-:|:-:|:-:|
| Greek | 5.8 min | 0.824 | 0.814 | 0.818 | 0.986 |
| Arabic | 5.4 min | 0.835 | 0.863 | 0.868 | 0.989 |
| Chinese | 5.1 min | 0.593 | 0.597 | 0.607 | 0.990 |
| Cyrillic | 19.3 min | 0.736 | 0.731 | 0.738 | 0.984 |
| Latin | 91.3 min | 0.799 | 0.796 | 0.798 | 0.996 |

Source: `results/runs/*/metrics.json`. These are single-seed results.

## 5. Problems found and how they were solved

| Problem | Solution |
|:-|:-|
| Results changed by up to 15 points between identical runs (no seed was ever set) | Seeds are set everywhere; `config.json` records them; seeds 43 and 44 are planned |
| Best epoch was chosen on the test split | Selection uses validation only; the test set is read once, with a test that enforces it |
| Each family saved its own head, but the old app loaded them all into one shared head, so only the last survived | Heads and adapters are kept per family and activated together at inference (Member 3) |
| The contrastive loss turned into NaN when a batch had no matching pair | `torch.where` masking and a zero contribution for such rows |
| PEFT crashed on Kaggle (`torchao` too old) | `compat.py` |
| Kaggle sessions can be wiped mid-run, and Latin alone takes about 90 minutes | Per-epoch resume with exact parity, plus `DONE.json` to skip finished runs |
| Claims in the README had no source | Every metric now comes from `metrics.json`; unverified claims were marked |

## 6. How to run and verify

```bash
python -m pytest 02_model_training/tests -q                       # 10 CPU tests
python 02_model_training/src/train.py --cluster greek --smoke     # smoke test, seconds
bash 02_model_training/scripts/train_all_local.sh                 # the real runs, about 2 hours on a GPU
```
Use `--shuffle_labels` on a small family: the score must drop to about 1/8. If it does not, there is a leak.

## 7. Limits

- One seed per family. Differences of a few points between runs are within the noise seen before.
- Chinese is the weakest family (macro-F1 0.60); naming the exact generator is much harder than telling AI from human.
- The claim that the loss pulls generators together **across scripts** is not tested; training is within one family.
- No baseline (one adapter for everything) or ablation has been run yet; they are in the next-work list.

## 8. Viva questions

1. **Why freeze the backbone?** It keeps the multilingual knowledge, makes training fast and small, and lets one backbone serve five adapters.
2. **What is LoRA?** Two thin matrices added beside a large weight matrix; only they train, so about 0.3 % of the parameters change.
3. **What does the contrastive loss add?** It makes the internal representation cluster by generator, which helps tell generators apart.
4. **How do you know the test score is honest?** Epochs are picked on validation, the test split is read once, and a test enforces this.
5. **Why is Chinese worse?** It has few examples, a different script, and few human test documents; this is an open question.
