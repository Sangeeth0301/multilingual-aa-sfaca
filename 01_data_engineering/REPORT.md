# Member 1: Data engineering

## 1. Role and objective

**Objective:** turn the raw MULTITuDE v3 file into data that can be trusted. Every number the project reports depends
on this folder. If the data leaks between train and test, or if the labels are fake, every later result is inflated.

The raw file has 206,330 rows. The work in this folder:

1. Groups the 21 languages into **five script families** (Latin, Cyrillic, Greek, Arabic, Chinese).
2. Removes data that must not be used (fake rows, duplicates, three held-out languages).
3. Builds **train / validation / test splits that cannot leak** and freezes them with a hash.
4. Loads the result into PyTorch for training and evaluation.
5. Documents the data in a **data card**, with a reproducibility check.

## 2. What is in this folder

| File | Lines | Purpose |
|:-|:-:|:-|
| `src/clusters.py` | 136 | Language to script-family mapping used by the whole project |
| `src/data_loader.py` | 263 | Loads the CSV and splits safely, class labels, sampling, PyTorch dataset |
| `scripts/inspect_data.py` | 235 | First audit of the raw file |
| `scripts/build_splits.py` | 400 | Builds the leak-free splits |
| `scripts/add_flags_and_similarity.py` | 98 | Adds quality flags and a similarity measure |
| `scripts/check_leakage.py` | 104 | Fails loudly if any leak exists |
| `scripts/cross_split_similarity.py` | 178 | Measures how similar validation and test are to train |
| `scripts/group_structure.py` | 115 | Reports the size and balance of story groups |
| `scripts/verify_reproducible.py` | 80 | Proves the splits can be rebuilt identically |
| `tests/test_check_leakage.py` | 105 | Automated tests for the leakage checker |
| `DATA_CARD.md` | 191 | The written description of the dataset |

## 3. How the code works, file by file

### `src/clusters.py`: the core design choice
**Objective:** decide which languages share an adapter.
**Logic:** `SCRIPT_CLUSTERS` is a dictionary from cluster name to languages, for example Latin has `nl, en, de, pt, ro, es,
hr, cs, pl, sk, sl, hu`, Cyrillic has `bg, uk, ru`, Greek `el`, Arabic `ar`, Chinese `zh`. From it the file builds
`LANG_TO_CLUSTER`. `BASE_PAPER_18_LANGUAGES` is the list used in the original research paper so results stay comparable.
Catalan, Irish and Scottish Gaelic (`ca, ga, gd`) are in the raw file but are kept out of training; they are reserved for
zero-shot testing.
**Key functions:** `get_cluster(lang)` returns the family, `all_clusters()` lists them, and
`nearest_cluster_for_unseen(lang)` picks the closest family for a language that was never seen.

### `src/data_loader.py`: safe loading
**Objective:** give the training code the data in one correct, verified table.
**Logic:**
- `LABEL_LIST_8CLASS` fixes the class order: human plus seven generators. The old 10-class dictionary is kept only so old
  files still read.
- `load_training_table(raw_path, splits_path)` first computes the SHA-256 of the splits file and **refuses to run if it
  differs from the frozen value** `8c05df83…` unless `allow_new_splits=True`. It then joins the raw text to the splits on
  `row_id` (the raw file's `Unnamed: 0` column), checks that no rows were lost, and maps every `multi_label` to a number,
  raising an error on any unknown label instead of training on broken labels.
- `stratified_subsample` caps the number of training rows deterministically and keeps the classes balanced, so a capped run
  can be repeated exactly.
- `AttributionDataset` is a thin PyTorch `Dataset`. For each row it tokenizes the text on the fly (truncated and padded to
  `max_length`, 128 by default) and returns the token ids, attention mask, label, cluster and row id.

### `scripts/inspect_data.py`: the first audit
**Objective:** look at the raw data before changing anything.
**Logic:** it reports columns, text lengths, the `source` column, duplicates, the Gemini/Claude rows and the class balance,
and writes `results/data_inspection.md` and `.json`. This audit found that the Gemini and Claude "classes" were not real
(see section 5).

### `scripts/build_splits.py`: the leak-free splits
**Objective:** make sure no story appears in two splits.
**Logic, in order:**
1. Mark junk rows and rows in the wrong script.
2. Keep only the 18 base-paper languages; set `ca, ga, gd` aside in `results/ca_ga_gd_ids.csv`.
3. Drop the template-based Gemini and Claude rows.
4. Remove any training text whose normalized hash appears in the original test set, then remove duplicates inside train.
5. **Group stories.** MULTITuDE has one human article and several machine rewrites of the same story. For each
   (language, outlet, generator) bucket the script computes TF-IDF cosine similarity between human and machine texts and
   solves a one-to-one assignment (`scipy.optimize.linear_sum_assignment`, the Hungarian method) so each human text gets at
   most one machine text per generator. Leftover texts are attached greedily to their best match.
6. Carve a validation set from train with `StratifiedGroupKFold`, which keeps a whole group inside one split and the class
   mix balanced.
7. The original test rows stay exactly as they were (`group_id = -1`).
**Output:** `data/processed/splits_8class.csv` and `results/split_report.md`.

### `scripts/add_flags_and_similarity.py`
**Objective:** mark weak rows so they can be excluded from a clean evaluation.
**Logic:** `flag_junk` is script-aware (fewer than 20 letters for most scripts, fewer than 5 ideographs for Chinese);
`flag_script_mismatch`, `flag_prompt_echo` (text that leaks the prompt such as "task write"); and `story_disjoint`, true when
a row's highest similarity to opposite-label training text is below 0.7. These columns allow extra "clean subset" scores.

### `scripts/check_leakage.py`: the gatekeeper
**Objective:** stop the project if the data is unsafe. It exits with code 1 on any failure.
**Five checks:** (1) no `ca/ga/gd` rows anywhere; (2) no Gemini/Claude labels anywhere; (3) test rows are exactly the
original test rows of the 18 languages; (4) no normalized-text hash is shared across train, validation and test; (5) no
`group_id` appears in more than one split.

### `scripts/cross_split_similarity.py` and `scripts/group_structure.py`
**Objective:** measure how safe the splits really are. The first compares validation-to-train and test-to-train TF-IDF
similarity against a random control, and found that validation is somewhat leakier than test. The second reports group
sizes (median 8, maximum 11), which confirmed the earlier hub-group problem is gone.

### `scripts/verify_reproducible.py`
**Objective:** prove the pipeline is deterministic. It runs `build_splits` and `add_flags_and_similarity` and compares the
hashes of the outputs.

### `tests/test_check_leakage.py`
Five tests on small synthetic tables: a clean table passes; a text-hash overlap fails; a story (`group_id`) shared across
splits fails; a Gemini label fails; and an empty-text hash is correctly ignored.

## 4. Inputs and outputs

| | Item |
|:-|:-|
| **Reads** | `data/raw/multitude_v3_clean.csv` (SHA-256 `1e08f9c6…`) |
| **Writes** | `data/processed/splits_8class.csv` (SHA-256 `8c05df83…`), `results/*.md` data reports |
| **Hands over** | `load_training_table()`, `LABEL_LIST_8CLASS`, `get_cluster()`, the frozen splits file |

## 5. Problems found and how they were solved

| Problem | How it was solved |
|:-|:-|
| The Gemini and Claude classes were about 30 hand-written English sentences repeated about 1,800 times, in both train and test, so any result with them was inflated | Dropped from every training and evaluation path; the task became 8 classes |
| The test split had been used as validation, so the best epoch was chosen on test | A real validation split was carved from train, grouped by story |
| Earlier grouping created "hub" groups of up to 821 rows | One-to-one assignment; maximum group size is now 11 |
| The same text could sit in train and test | Hash purge plus `check_leakage.py` |
| The splits could silently change between runs | Frozen SHA-256, checked on every training run |
| Chinese text looked "junk" under a letter count | A script-aware junk rule |

## 6. Evidence

- 186,008 rows after filtering; 18 languages; 8 classes. See `DATA_CARD.md`.
- Raw and splits file hashes from the Kaggle download matched the data card.
- `check_leakage.py` passes; 47 project tests pass from the repository root.

## 7. How to run and verify

```bash
python 01_data_engineering/scripts/check_leakage.py          # must exit 0
python 01_data_engineering/scripts/inspect_data.py           # data audit (overwrites results/data_inspection.*)
python -m pytest 01_data_engineering/tests -q
```
Rebuilding the splits (`build_splits.py`) changes the frozen hash and invalidates every trained result, so do it only on purpose.

## 8. Limits

- Validation is leakier than test (documented in the data card), so validation scores are slightly optimistic.
- The data is news text from 2023-era generators. It has no newer generators, no essays and no social media.
- Chinese has few human test documents.

## 9. Viva questions

1. **Why are splits grouped by story?** Machine rewrites of one article look alike; without grouping, a model can pass by memorizing a story.
2. **Why freeze the splits with a hash?** So every trained model, by anyone, was tested on exactly the same rows.
3. **What was wrong with the Gemini and Claude classes?** They were templates, not real model output, and appeared in train and test.
4. **Why are `ca`, `ga` and `gd` excluded?** They are held out to test transfer to unseen languages.
5. **What does `check_leakage.py` prove, and what does it not?** It proves no text or story crosses splits; it cannot prove the generators are representative of new models.
