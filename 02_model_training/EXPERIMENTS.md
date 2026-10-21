# SFA-CA Experiment Execution Protocol (Phase C)

This document contains the exact reproducible CLI commands for training the SFA-CA model suite on Kaggle (Tesla T4 GPU), ordered strictly by priority.

All runs use frozen split verification (`splits_8class.csv` SHA-256: `8c05df8355c44611b05aa290c6644f822436734a74da278f65da1c91f370cf0b`), deterministic seeding, epoch selection on **val macro-F1 only**, and test split evaluation **once** at the end.

---

## 1. Common Settings & Hyperparameters

Reused from original `train.py` and validated on Kaggle T4:
- **Optimizer**: `AdamW(lr=2e-4)`
- **Learning Rate Schedule**: `none` (constant learning rate, default)
- **Warmup Steps**: `0`
- **Gradient Clipping**: `none` (`max_grad_norm=0.0`)
- **Batch Size**: `32`
- **Max Sequence Length**: `128`
- **Epochs**: `3`
- **Mixed Precision**: FP16 Automatic Mixed Precision (AMP)
- **LoRA Configuration**: `r=8`, `alpha=16`, `dropout=0.05` on target modules `query` and `value` (XLM-RoBERTa attention projections)
- **Contrastive Objective**: Supervised contrastive loss weight `0.5`, temperature `0.1` (within-cluster in single cluster runs; cross-script in monolithic baseline)

---

## 2. Estimated GPU Runtime (Tesla T4)

> [!NOTE]
> Timings are **ESTIMATES** derived from `KAGGLE_RUN_FACTS.md` (~55 samples/second on T4 with batch size 32, max length 128, and FP16 AMP) applied to the official 8-class train split sizes.

| Cluster | Train Rows | 3-Epoch Total Samples | Estimated Time per Run |
|:---|:---:|:---:|:---:|
| **Arabic** | 7,178 | 21,534 | **~6.5 min** |
| **Greek** | 7,159 | 21,477 | **~6.5 min** |
| **Hanzi** | 7,126 | 21,378 | **~6.5 min** |
| **Cyrillic** | 21,458 | 64,374 | **~19.5 min** |
| **Latin** | 85,839 | 257,517 | **~78.0 min (1.30 hrs)** |
| **Monolithic Baseline (`all`)** | 128,760 | 386,280 | **~117.1 min (1.95 hrs)** |

---

## 3. Ordered Experiment Matrix

### Phase C.1: Smoke Test & Sanity Checks

#### A. Smoke Test (Arabic, 1 Epoch, 500 Samples)
*Budget validation: prints exact time per 1,000 samples on the host GPU.*
```bash
python 02_model_training/src/train.py \
  --cluster arabic \
  --seed 42 \
  --smoke \
  --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv \
  --splits_data /kaggle/input/sfaca-data/splits_8class.csv \
  --out_dir /kaggle/working/results/runs
```
*Estimated time: ~10 seconds.*

#### B. Shuffled-Label Sanity Check (Arabic and Greek)
*Integrity check: TRAIN labels are permuted while VAL/TEST labels remain true. If macro-F1 does not drop near chance (~0.125), stop and report leakage.*
```bash
# Arabic shuffled-label check
python 02_model_training/src/train.py \
  --cluster arabic \
  --seed 42 \
  --shuffle_labels \
  --run_tag shuffled \
  --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv \
  --splits_data /kaggle/input/sfaca-data/splits_8class.csv \
  --out_dir /kaggle/working/results/runs

# Greek shuffled-label check
python 02_model_training/src/train.py \
  --cluster greek \
  --seed 42 \
  --shuffle_labels \
  --run_tag shuffled \
  --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv \
  --splits_data /kaggle/input/sfaca-data/splits_8class.csv \
  --out_dir /kaggle/working/results/runs
```
*Estimated time: ~13 min total.*

---

### Phase C.2: Lean Plan (Core Attribution Matrix & Key Baselines)

Total GPU budget: **~5.5 hours** on T4.

#### 1. Arabic Adapters (3 Seeds)
```bash
python 02_model_training/src/train.py --cluster arabic --seed 42 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster arabic --seed 43 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster arabic --seed 44 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~19.5 min.*

#### 2. Greek Adapters (3 Seeds)
```bash
python 02_model_training/src/train.py --cluster greek --seed 42 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster greek --seed 43 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster greek --seed 44 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~19.5 min.*

#### 3. Hanzi Adapters (3 Seeds)
```bash
python 02_model_training/src/train.py --cluster hanzi --seed 42 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster hanzi --seed 43 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster hanzi --seed 44 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~19.5 min.*

#### 4. Cyrillic Adapters (3 Seeds)
```bash
python 02_model_training/src/train.py --cluster cyrillic --seed 42 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster cyrillic --seed 43 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster cyrillic --seed 44 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~58.5 min.*

#### 5. Latin Adapter (Seed 42)
```bash
python 02_model_training/src/train.py --cluster latin --seed 42 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~78.0 min.*

#### 6. Monolithic Baseline (`--cluster all`, Seed 42)
*Trained on the union of all 5 clusters. Batches mix scripts, resulting in cross-script contrastive pairs.*
```bash
python 02_model_training/src/train.py --cluster all --seed 42 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~117.1 min.*

#### 7. Contrastive Weight Ablation (Weight 0.0 vs 0.5)
```bash
# Arabic without contrastive loss
python 02_model_training/src/train.py --cluster arabic --seed 42 --contrastive_weight 0.0 --run_tag con0 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs

# Greek without contrastive loss
python 02_model_training/src/train.py --cluster greek --seed 42 --contrastive_weight 0.0 --run_tag con0 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~13.0 min.*

#### 8. LoRA Rank Ablation (Rank 16 and 32)
```bash
# Arabic LoRA r=16
python 02_model_training/src/train.py --cluster arabic --seed 42 --lora_r 16 --lora_alpha 32 --run_tag r16 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs

# Arabic LoRA r=32
python 02_model_training/src/train.py --cluster arabic --seed 42 --lora_r 32 --lora_alpha 64 --run_tag r32 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~13.0 min.*

---

### Phase C.3: Extension Plan (If Kaggle Quota Allows)

Total extra GPU budget: **~6.5 hours** on T4.

#### 9. Latin Multi-Seed (Seeds 43 & 44)
```bash
python 02_model_training/src/train.py --cluster latin --seed 43 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster latin --seed 44 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~156 min (~2.6 hrs).*

#### 10. Monolithic Baseline Multi-Seed (Seeds 43 & 44)
```bash
python 02_model_training/src/train.py --cluster all --seed 43 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
python 02_model_training/src/train.py --cluster all --seed 44 --raw_data /kaggle/input/sfaca-data/multitude_v3_clean.csv --splits_data /kaggle/input/sfaca-data/splits_8class.csv --out_dir /kaggle/working/results/runs
```
*Estimated time: ~234 min (~3.9 hrs).*
