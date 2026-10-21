#!/usr/bin/env bash
# Train all 5 clusters sequentially (smallest first). Resumable: finished runs have DONE.json.
cd "$(dirname "$0")/../.."
for c in greek arabic hanzi cyrillic latin; do
  python 02_model_training/src/train.py --cluster "$c" --seed 42 --epochs 3 --batch_size 16 --max_length 128 \
    --lr 2e-4 --out_dir results/runs 2>&1 | grep -v -i "warning\|LOAD REPORT\|UNEXPECTED" > "results/train_${c}.log"
done
echo ALL_DONE > results/train_all.done
