# KAGGLE RUN FACTS (verified from `sfaca_kaggle_original_REDACTED.ipynb`)

Put this file in `notebooks/kaggle/` next to the redacted notebook. Everything below was read from the notebook's code and printed outputs. Anything marked UNVERIFIED could not be found there.

## Environment
- GPU: Tesla T4 (Kaggle). Python 3.12.
- Packages were NOT pinned (`transformers>=4.40`, `peft>=0.10`...). Later cells pin `peft==0.13.2` and install `torchao>=0.16`; several cells monkey-patch `peft.tuners.lora.torchao` to avoid import errors. Exact torch/transformers versions: UNVERIFIED (get `pip freeze` from Kaggle).
- Data: `multitude_v3_clean.csv`, shape (206330, 8), columns `Unnamed: 0, text, label, multi_label, split, language, length, source`. `source` is probably the source dataset, not an article id (verify before using it for grouped splits).
- Experiment tracking: W&B project `sfa-ca-multilingual-attribution` (run configs are the authoritative record of hyperparameters).

## Hyperparameters actually used (every training cell)
`--epochs 3 --batch_size 32 --max_length 128 --lr 2e-4`, contrastive weight default 0.5, LoRA r=8 alpha=16 dropout 0.05 on query/value. NO seed set. Best epoch chosen on the TEST split.
Therefore the inference `max_length` must be 128 (not 256).

## Training sample counts
Clean 8-class data: latin 95,431 train / 28,631 test; cyrillic 23,838 / 7,153; greek 7,944 / 2,384; arabic 7,975 / 2,392; hanzi 7,926 / 2,383.
10-class (template) data: latin 102,631 / 35,831; cyrillic 29,238 / 12,553; greek 9,744 / 4,184; arabic 9,775 / 4,192; hanzi 9,726 / 4,183.

## Measured per-cluster macro-F1 (selected on test split)
| Run | Data | Latin | Cyrillic | Greek | Arabic | Hanzi |
|:-|:-|:-:|:-:|:-:|:-:|:-:|
| First clean runs (cells 11-21) | 8-class | crashed / 0.7733 (cell 23 rerun) | 0.7743 | 0.7853 | 0.8762 | 0.6696 |
| Re-run clean (cells 32-33 matrix diagonal) | 8-class | 0.7448 | 0.7634 | 0.6314 | 0.7981 | 0.6008 |
| Early 10-class (cell 38) | 10-class v1 | 0.8484 | 0.8000 | 0.7719 | 0.8364 | n/a |
| Final 10-class (cells 53-56 diagonal) | 10-class templates | 0.8105 | 0.7885 | 0.7750 | 0.8684 | 0.6405 |

Same config, different runs, differences up to 15 points (Greek 0.7853 vs 0.6314). Single-run numbers are not reliable.

## Where the README numbers come from
- README diagonal 0.8762 / 0.7853 / 0.7743 / 0.6696: real, from the FIRST clean 8-class runs (cells 19, 16, 14, 21). README Latin 0.7627: not found (nearest 0.7733, 0.7448).
- README off-diagonal values (20 numbers), "+19.46%", "<27 ms": NOT found in any output. UNVERIFIED.
- Matrix actually computed on clean data (cell 33, rows = train adapter): arabic [ar .798, cy .309, el .497, zh .253, la .255]; cyrillic [.615, .763, .607, .391, .605]; greek [.475, .376, .631, .245, .330]; hanzi [.213, .211, .170, .601, .167]; latin [.555, .665, .577, .546, .745]. (column order arabic, cyrillic, greek, hanzi, latin as printed in the notebook)
- Matrix computed on 10-class template data (cell 56): latin [.8105 .7562 .6708 .6539 .6638] (la,cy,el,ar,zh); cyrillic [.6842 .7885 .5961 .6300 .5187]; greek [.4357 .4841 .7750 .5765 .4191]; arabic [.4978 .5318 .6264 .8684 .4429]; hanzi [.3280 .3663 .3285 .3751 .6405].

## Zero-shot "low resource" (cell 37, Latin adapter, clean 8-class adapters at the time)
Catalan `ca` F1 0.7432, Irish `ga` 0.4160, Scots Gaelic `gd` 0.3169. Tamil `ta`: no data, a DUMMY suite was created (cell 36). No Indic or African results exist.

## Timings measured from W&B run timestamps (T4, bs 32, len 128, 3 epochs)
Latin about 87 min per run; Cyrillic about 22 min; Greek/Arabic/Hanzi (about 8k samples) roughly 7-8 min each (estimate by sample ratio). A 3-seed run of all 5 clusters is roughly 7 hours of GPU time; the baseline on all clusters is extra. Budget against Kaggle's current weekly GPU quota. Cell comments claiming "~10 min total" are wrong.

## Problems seen in the notebook logs
- `evaluate_cross_cluster.py` failed many times with `HFValidationError` (it used `load_adapter` on a local path) before being fixed in the repo; early matrix attempts were incomplete (cell 28 says 16 of 25 pairs).
- API crashed with `KeyError` from `get_cluster(sanitized_text)` and with `RecursionError` during model creation until patched; the demo hit 500 errors for several versions.
- Multiple cells duplicate each other (training re-runs after session resets). Final adapters on disk: 10-class template-trained. The clean 8-class adapters were likely overwritten; confirm whether a backup exists.
- `evaluate_cross_cluster.py` was run on test splits of the same data used to choose epochs.

## Not in this notebook
No monolithic baseline, no plain RoBERTa/XLM-R full fine-tune, no latency benchmark, no calibration, no sanitizer evaluation. If the human ran such notebooks elsewhere, they must be added to `notebooks/kaggle/` and audited.
