# MULTITuDE V3 Clean - Data Card

## Dataset Facts
- **Source**: `data/raw/multitude_v3_clean.csv`
- **Total Rows**: 186,008
- **Task**: 8-class AI text attribution (1 human class, 7 AI generator classes)
- **SHA-256 (Raw CSV)**: 1E08F9C6980ED9CAE11BC831D8CE54D13125EA9A52DFEF387B0B4EBD0909647C
- **SHA-256 (Splits File, full — all columns)**: 8C05DF8355C44611B05AA290C6644F822436734A74DA278F65DA1C91F370CF0B
- **SHA-256 (Splits File, frozen key cols: row_id, cluster, split, group_id)**: C71CDBC5817FE5229BD414A86E8D0C98D2E241A84D457BD43E144BE545904F71

> The key-columns hash covers only the four assignment columns and is frozen — adding derived columns (flags, similarity) does not change it.

## Languages
- **Included (18 Base-Paper Languages)**: ar, bg, cs, de, el, en, es, hr, hu, nl, pl, pt, ro, ru, sk, sl, uk, zh
- **Excluded Set**: ca, ga, gd — removed from train/val/test; kept strictly for zero-shot evaluation only.

## Splits Method and Limitations
Train and val split assignments were generated via a one-to-one assignment (scipy `linear_sum_assignment`). Each human text receives at most one machine text per generator, maximizing total TF-IDF cosine similarity. This prevents the formation of massive hub groups that occurred during an earlier heuristic grouping phase.

- **Validation Leakage**: In most languages, the val set is leakier than the test set (`val->train` similarity > `test->train` similarity).
- **Test rows**: All original MULTITuDE test rows were kept unchanged and were **not** regrouped; they carry `group_id = -1`.

## Group Statistics (Train + Val Groups Only)

*Computed by `scripts/report_stats.py`*

- **Human anchors**: 17992
- **Group sizes**: min = 2 · median = 8.0 · p90 = 8.0 · max = 11

> **Note on previous grouping**: An earlier heuristic best-match grouping produced hub groups (up to 821 rows). The cause was not investigated. The current one-to-one assignment fixes this issue.

### Top 10 Largest Groups

| Rank | group_id | Size | Language | Outlet | Split(s) |
|------|----------|------|----------|--------|----------|
| 1 | 4692 | 11 | el | MassiveSumm_tovima | train |
| 2 | 16837 | 9 | uk | MassiveSumm_unian | train |
| 3 | 4621 | 9 | el | MassiveSumm_tanea | train |
| 4 | 16763 | 9 | uk | MassiveSumm_unian | train |
| 5 | 16904 | 9 | uk | MassiveSumm_unian | train |
| 6 | 16756 | 9 | uk | MassiveSumm_unian | train |
| 7 | 16733 | 9 | uk | MassiveSumm_unian | train |
| 8 | 16725 | 9 | uk | MassiveSumm_unian | train |
| 9 | 16717 | 9 | uk | MassiveSumm_unian | train |
| 10 | 16696 | 9 | uk | MassiveSumm_unian | train |

## Flag Definitions

- `flag_junk` (script-aware):
  - Latin, Cyrillic, Greek, Arabic clusters: True if the text contains **fewer than 20 letter characters**.
  - Hanzi cluster: True if the text contains **fewer than 5 CJK ideographs or Kana characters**.
- `flag_script_mismatch`: True if the text contains mostly out-of-cluster characters for its designated script cluster.
- `flag_prompt_echo`: True if the text matches known prompt-leakage regex patterns (e.g. `(?i)\b(task write|task translate)\b`).
- `story_disjoint`: True if `max_sim_train_other < 0.7` (the row's max TF-IDF cosine similarity to train rows of the opposite label in the same language).
- `max_sim_train_other`: Populated for val and test only; empty for train.

## Junk Counts (Script-Aware, After Fix)

| cluster | split | generator | count |
|---------|-------|-----------|-------|
| arabic | train | Llama-2-70b-chat-hf | 1 |
| arabic | train | human | 5 |
| arabic | train | opt-iml-max-30b | 1 |
| arabic | train | vicuna-13b | 1 |
| cyrillic | train | human | 2 |
| cyrillic | train | opt-iml-max-30b | 2 |
| greek | train | opt-iml-max-30b | 1 |
| hanzi | train | Llama-2-70b-chat-hf | 78 |
| hanzi | train | Mistral-7B-Instruct-v0.2 | 12 |
| hanzi | train | opt-iml-max-30b | 2 |
| hanzi | train | v5-Eagle-7B-HF | 12 |
| hanzi | train | vicuna-13b | 19 |
| latin | train | human | 16 |
| latin | train | opt-iml-max-30b | 15 |
| latin | train | vicuna-13b | 1 |
| hanzi | val | Llama-2-70b-chat-hf | 14 |
| hanzi | val | Mistral-7B-Instruct-v0.2 | 2 |
| hanzi | val | v5-Eagle-7B-HF | 1 |
| hanzi | val | vicuna-13b | 3 |
| latin | val | human | 2 |
| latin | val | opt-iml-max-30b | 6 |
| latin | val | v5-Eagle-7B-HF | 1 |
| cyrillic | test | opt-iml-max-30b | 1 |
| hanzi | test | Llama-2-70b-chat-hf | 26 |
| hanzi | test | Mistral-7B-Instruct-v0.2 | 6 |
| hanzi | test | v5-Eagle-7B-HF | 4 |
| hanzi | test | vicuna-13b | 7 |
| latin | test | Mistral-7B-Instruct-v0.2 | 1 |
| latin | test | human | 5 |
| latin | test | opt-iml-max-30b | 25 |

> Human junk rows are short/empty originals in the raw dataset. Hanzi Llama-2-70b is the worst offender (the model emitted English explanations instead of Chinese text). The hanzi human train count is **0** (none of the real Chinese human articles are flagged under the CJK-aware threshold).

## Story Disjointness

The `max_sim_train_other` column (val/test rows only) is the maximum TF-IDF cosine similarity of a row to any train row of a **different** label in the same language. `story_disjoint = (max_sim_train_other < 0.7)`.

| Language | Val % disjoint | Test % disjoint |
|----------|---------------|-----------------|
| ar | 97.9 | 97.8 |
| bg | 98.0 | 97.4 |
| cs | 98.2 | 97.8 |
| de | 98.6 | 99.4 |
| el | 97.8 | 97.7 |
| en | 94.8 | 97.3 |
| es | 97.7 | 98.5 |
| hr | 99.1 | 98.6 |
| hu | 98.7 | 98.7 |
| nl | 98.1 | 99.0 |
| pl | 98.2 | 98.3 |
| pt | 97.3 | 97.4 |
| ro | 98.5 | 98.1 |
| ru | 96.6 | 97.6 |
| sk | 97.4 | 97.8 |
| sl | 98.8 | 99.1 |
| uk | 96.6 | 98.0 |
| zh | 91.4 | 95.3 |

## Final Counts per Cluster × Split × Class

| Cluster | Split | Max/Min Ratio | Counts |
|---------|-------|---------------|--------|
| arabic | train | 1.01 | gpt-3.5-turbo-0125: 900, human: 900, aya-101: 900, Mistral-7B-Instruct-v0.2: 899, v5-Eagle-7B-HF: 899, vicuna-13b: 898, opt-iml-max-30b: 894, Llama-2-70b-chat-hf: 888 |
| arabic | val | 1.02 | aya-101: 100, v5-Eagle-7B-HF: 100, gpt-3.5-turbo-0125: 100, Mistral-7B-Instruct-v0.2: 100, vicuna-13b: 100, human: 100, Llama-2-70b-chat-hf: 99, opt-iml-max-30b: 98 |
| arabic | test | 1.02 | Mistral-7B-Instruct-v0.2: 300, aya-101: 300, human: 300, gpt-3.5-turbo-0125: 300, v5-Eagle-7B-HF: 300, vicuna-13b: 299, opt-iml-max-30b: 298, Llama-2-70b-chat-hf: 295 |
| cyrillic | train | 1.03 | aya-101: 2700, Mistral-7B-Instruct-v0.2: 2699, gpt-3.5-turbo-0125: 2699, human: 2696, vicuna-13b: 2693, v5-Eagle-7B-HF: 2692, opt-iml-max-30b: 2664, Llama-2-70b-chat-hf: 2615 |
| cyrillic | val | 1.04 | Mistral-7B-Instruct-v0.2: 300, gpt-3.5-turbo-0125: 300, human: 300, aya-101: 300, vicuna-13b: 299, v5-Eagle-7B-HF: 299, opt-iml-max-30b: 294, Llama-2-70b-chat-hf: 288 |
| cyrillic | test | 1.03 | aya-101: 900, gpt-3.5-turbo-0125: 900, Mistral-7B-Instruct-v0.2: 899, vicuna-13b: 899, human: 898, v5-Eagle-7B-HF: 896, opt-iml-max-30b: 890, Llama-2-70b-chat-hf: 871 |
| greek | train | 1.03 | aya-101: 901, gpt-3.5-turbo-0125: 900, Mistral-7B-Instruct-v0.2: 899, human: 899, v5-Eagle-7B-HF: 899, vicuna-13b: 895, opt-iml-max-30b: 888, Llama-2-70b-chat-hf: 878 |
| greek | val | 1.03 | human: 99, aya-101: 99, gpt-3.5-turbo-0125: 99, Mistral-7B-Instruct-v0.2: 99, v5-Eagle-7B-HF: 98, vicuna-13b: 98, Llama-2-70b-chat-hf: 96, opt-iml-max-30b: 96 |
| greek | test | 1.02 | aya-101: 300, human: 300, gpt-3.5-turbo-0125: 300, v5-Eagle-7B-HF: 299, Mistral-7B-Instruct-v0.2: 299, Llama-2-70b-chat-hf: 296, vicuna-13b: 295, opt-iml-max-30b: 295 |
| hanzi | train | 1.04 | human: 899, aya-101: 899, Mistral-7B-Instruct-v0.2: 898, v5-Eagle-7B-HF: 896, opt-iml-max-30b: 892, vicuna-13b: 889, gpt-3.5-turbo-0125: 888, Llama-2-70b-chat-hf: 865 |
| hanzi | val | 1.04 | Mistral-7B-Instruct-v0.2: 101, aya-101: 101, human: 101, v5-Eagle-7B-HF: 101, vicuna-13b: 100, opt-iml-max-30b: 100, gpt-3.5-turbo-0125: 99, Llama-2-70b-chat-hf: 97 |
| hanzi | test | 1.02 | human: 300, Mistral-7B-Instruct-v0.2: 300, aya-101: 300, v5-Eagle-7B-HF: 299, opt-iml-max-30b: 297, Llama-2-70b-chat-hf: 296, vicuna-13b: 296, gpt-3.5-turbo-0125: 295 |
| latin | train | 1.03 | gpt-3.5-turbo-0125: 10799, human: 10798, Mistral-7B-Instruct-v0.2: 10795, aya-101: 10794, vicuna-13b: 10784, v5-Eagle-7B-HF: 10784, opt-iml-max-30b: 10585, Llama-2-70b-chat-hf: 10500 |
| latin | val | 1.03 | human: 1200, aya-101: 1200, vicuna-13b: 1200, gpt-3.5-turbo-0125: 1199, Mistral-7B-Instruct-v0.2: 1198, v5-Eagle-7B-HF: 1197, opt-iml-max-30b: 1184, Llama-2-70b-chat-hf: 1166 |
| latin | test | 1.03 | gpt-3.5-turbo-0125: 3600, Mistral-7B-Instruct-v0.2: 3600, human: 3599, aya-101: 3598, v5-Eagle-7B-HF: 3592, vicuna-13b: 3591, opt-iml-max-30b: 3550, Llama-2-70b-chat-hf: 3501 |

## Shortcut Baselines

Macro-F1 (chance ≈ 0.125) on val and test sets using simple shortcut features.

| Cluster | Model | Val F1 | Test F1 | Test F1 (Clean Subset) |
|---|---|---|---|---|
| cyrillic | (a) Length Features | 0.291 | 0.292 | 0.293 |
| cyrillic | (b) First 3 Words TF-IDF | 0.322 | 0.313 | 0.308 |
| cyrillic | (c) Flags Only | 0.067 | 0.059 | 0.029 |
| cyrillic | (d) All Combined | 0.343 | 0.332 | 0.315 |
| latin | (a) Length Features | 0.223 | 0.220 | 0.220 |
| latin | (b) First 3 Words TF-IDF | 0.309 | 0.303 | 0.295 |
| latin | (c) Flags Only | 0.051 | 0.050 | 0.028 |
| latin | (d) All Combined | 0.264 | 0.257 | 0.250 |
| greek | (a) Length Features | 0.318 | 0.325 | 0.324 |
| greek | (b) First 3 Words TF-IDF | 0.335 | 0.377 | 0.370 |
| greek | (c) Flags Only | 0.070 | 0.070 | 0.029 |
| greek | (d) All Combined | 0.436 | 0.423 | 0.395 |
| hanzi | (a) Length Features | 0.175 | 0.177 | 0.154 |
| hanzi | (b) First 3 Words TF-IDF | 0.394 | 0.383 | 0.365 |
| hanzi | (c) Flags Only | 0.086 | 0.073 | 0.030 |
| hanzi | (d) All Combined | 0.353 | 0.339 | 0.323 |
| arabic | (a) Length Features | 0.299 | 0.325 | 0.324 |
| arabic | (b) First 3 Words TF-IDF | 0.422 | 0.430 | 0.430 |
| arabic | (c) Flags Only | 0.057 | 0.051 | 0.029 |
| arabic | (d) All Combined | 0.401 | 0.420 | 0.419 |

## Regeneration Commands

Full pipeline from raw CSV only. Run in order; each step depends on the previous.

```bash
# 1. Build story groups and train/val/test split assignments
python scripts/build_splits.py

# 2. Add script-aware flag_junk, max_sim_train_other, story_disjoint columns
python scripts/add_flags_and_similarity.py

# 3. Verify leakage invariants
python scripts/check_leakage.py

# 4. Compute cross-split similarity report
python scripts/cross_split_similarity.py

# 5. Compute group stats and write data_card_metrics.md/json
python scripts/report_stats.py

# 6. Run shortcut baselines
python scripts/shortcut_baselines.py

# 7. Generate this data card
python scripts/generate_data_card.py
```
