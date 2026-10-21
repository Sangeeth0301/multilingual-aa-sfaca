import json
import pandas as pd
import numpy as np
import hashlib
import os

def get_file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest().upper()

def get_cols_hash(df, cols):
    df_sorted = df.sort_values('row_id').reset_index(drop=True)
    return hashlib.sha256(df_sorted[cols].to_csv(index=False).encode('utf-8')).hexdigest().upper()

def main():
    splits_path = 'data/processed/splits_8class.csv'
    metrics_path = 'results/data_card_metrics.json'
    
    if not os.path.exists(splits_path) or not os.path.exists(metrics_path):
        print("Missing required files.")
        return
        
    df = pd.read_csv(splits_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    full_sha = get_file_hash(splits_path)
    key_sha = get_cols_hash(df, ['row_id', 'cluster', 'split', 'group_id'])
    
    # Generate Top 10 Largest Groups table dynamically
    tv = df[(df['split'].isin(['train', 'val'])) & (df['group_id'] != -1)]
    gs = tv.groupby('group_id').size()
    large_groups = gs.sort_values(ascending=False).head(10)
    
    # We need 'source' to show Outlet. But 'source' isn't in splits_8class.csv
    # So we'll read raw to merge it
    raw = pd.read_csv('data/raw/multitude_v3_clean.csv')
    if 'Unnamed: 0' in raw.columns: raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else: raw['row_id'] = raw.index
    
    df_with_src = df.merge(raw[['row_id', 'source']], on='row_id', how='left')
    tv_src = df_with_src[(df_with_src['split'].isin(['train', 'val'])) & (df_with_src['group_id'] != -1)]
    
    top10_rows = []
    rank = 1
    for gid, sz in large_groups.items():
        gdf = tv_src[tv_src['group_id'] == gid]
        lang = gdf['language'].iloc[0]
        outlets = ', '.join(gdf['source'].astype(str).str.replace('MULTITuDE_','').unique())
        splits_str = ', '.join(gdf['split'].unique())
        top10_rows.append(f"| {rank} | {gid} | {sz} | {lang} | {outlets} | {splits_str} |")
        rank += 1
        
    top10_str = '\n'.join(top10_rows)
    
    # Story disjoint
    sd_rows = []
    for lang in sorted(df['language'].unique()):
        s_val = df[(df['split'] == 'val') & (df['language'] == lang)]['story_disjoint'].mean() * 100
        s_test = df[(df['split'] == 'test') & (df['language'] == lang)]['story_disjoint'].mean() * 100
        if not pd.isna(s_val) or not pd.isna(s_test):
            v_str = f'{s_val:.1f}' if not pd.isna(s_val) else 'N/A'
            t_str = f'{s_test:.1f}' if not pd.isna(s_test) else 'N/A'
            sd_rows.append(f'| {lang} | {v_str} | {t_str} |')
    sd_str = '\n'.join(sd_rows)
    
    # Final counts table (reconstructing from data_card_metrics.json isn't strictly necessary if it's already generated somewhere, but it's not. I'll load results/split_report.md maybe? No, let's just compute it)
    counts_rows = []
    for cluster in sorted(df['cluster'].dropna().unique()):
        for split in ['train', 'val', 'test']:
            c_df = df[(df['cluster'] == cluster) & (df['split'] == split)]
            if len(c_df) == 0: continue
            
            vc = c_df['multi_label'].value_counts()
            if len(vc) == 0: continue
            mx = vc.max()
            mn = vc.min()
            ratio = mx / mn if mn > 0 else 0
            
            counts_str = ", ".join([f"{k}: {v}" for k, v in vc.items()])
            counts_rows.append(f"| {cluster} | {split} | {ratio:.2f} | {counts_str} |")
    final_counts_str = '\n'.join(counts_rows)
    
    # Read shortcut baselines
    try:
        with open('results/shortcut_baselines.md', 'r') as f:
            baselines_text = f.read()
            baselines_table = re.search(r'(\| Cluster \| Model.*)', baselines_text, re.DOTALL)
            if baselines_table:
                baselines_table = baselines_table.group(1)
            else:
                baselines_table = ""
    except:
        baselines_table = ""
        
    # Read junk counts
    junk_rows = []
    for split in ['train', 'val', 'test']:
        for cl in sorted(df['cluster'].dropna().unique()):
            for gen in sorted(df['multi_label'].unique()):
                s = df[(df['split']==split) & (df['cluster']==cl) & (df['multi_label']==gen)]
                if len(s) == 0: continue
                j_cnt = s['flag_junk'].sum()
                if j_cnt > 0:
                    junk_rows.append(f'| {cl} | {split} | {gen} | {j_cnt} |')
    junk_str = '\n'.join(junk_rows)

    card = f"""# MULTITuDE V3 Clean - Data Card

## Dataset Facts
- **Source**: `data/raw/multitude_v3_clean.csv`
- **Total Rows**: {len(df):,}
- **Task**: 8-class AI text attribution (1 human class, 7 AI generator classes)
- **SHA-256 (Raw CSV)**: 1E08F9C6980ED9CAE11BC831D8CE54D13125EA9A52DFEF387B0B4EBD0909647C
- **SHA-256 (Splits File, full — all columns)**: {full_sha}
- **SHA-256 (Splits File, frozen key cols: row_id, cluster, split, group_id)**: {key_sha}

> The key-columns hash covers only the four assignment columns and is frozen — adding derived columns (flags, similarity) does not change it.

## Languages
- **Included (18 Base-Paper Languages)**: ar, bg, cs, de, el, en, es, hr, hu, nl, pl, pt, ro, ru, sk, sl, uk, zh
- **Excluded Set**: ca, ga, gd — removed from train/val/test; kept strictly for zero-shot evaluation only.

## Splits Method and Limitations
Train and val split assignments were generated via a one-to-one assignment (scipy `linear_sum_assignment`). Each human text receives at most one machine text per generator, maximizing total TF-IDF cosine similarity. This prevents the formation of massive hub groups that occurred during an earlier heuristic grouping phase.

- **Validation Leakage**: In most languages, the val set is leakier than the test set (`val->train` similarity > `test->train` similarity).
- **Test rows**: All original MULTITuDE test rows were kept unchanged and were **not** regrouped; they carry `group_id = -1`.

## Group Statistics (Train + Val Groups Only)

*Computed by `03_inference_evaluation/scripts/report_stats.py`*

- **Human anchors**: {metrics.get('human_anchors', 17992)}
- **Group sizes**: min = {gs.min()} · median = {gs.median():.1f} · p90 = {np.percentile(gs, 90):.1f} · max = {gs.max()}

> **Note on previous grouping**: An earlier heuristic best-match grouping produced hub groups (up to 821 rows). The cause was not investigated. The current one-to-one assignment fixes this issue.

### Top 10 Largest Groups

| Rank | group_id | Size | Language | Outlet | Split(s) |
|------|----------|------|----------|--------|----------|
{top10_str}

## Flag Definitions

- `flag_junk` (script-aware):
  - Latin, Cyrillic, Greek, Arabic clusters: True if the text contains **fewer than 20 letter characters**.
  - Hanzi cluster: True if the text contains **fewer than 5 CJK ideographs or Kana characters**.
- `flag_script_mismatch`: True if the text contains mostly out-of-cluster characters for its designated script cluster.
- `flag_prompt_echo`: True if the text matches known prompt-leakage regex patterns (e.g. `(?i)\\b(task write|task translate)\\b`).
- `story_disjoint`: True if `max_sim_train_other < 0.7` (the row's max TF-IDF cosine similarity to train rows of the opposite label in the same language).
- `max_sim_train_other`: Populated for val and test only; empty for train.

## Junk Counts (Script-Aware, After Fix)

| cluster | split | generator | count |
|---------|-------|-----------|-------|
{junk_str}

> Human junk rows are short/empty originals in the raw dataset. Hanzi Llama-2-70b is the worst offender (the model emitted English explanations instead of Chinese text). The hanzi human train count is **0** (none of the real Chinese human articles are flagged under the CJK-aware threshold).

## Story Disjointness

The `max_sim_train_other` column (val/test rows only) is the maximum TF-IDF cosine similarity of a row to any train row of a **different** label in the same language. `story_disjoint = (max_sim_train_other < 0.7)`.

| Language | Val % disjoint | Test % disjoint |
|----------|---------------|-----------------|
{sd_str}

## Final Counts per Cluster × Split × Class

| Cluster | Split | Max/Min Ratio | Counts |
|---------|-------|---------------|--------|
{final_counts_str}

## Shortcut Baselines

Macro-F1 (chance ≈ 0.125) on val and test sets using simple shortcut features.

{baselines_table}

## Regeneration Commands

Full pipeline from raw CSV only. Run in order; each step depends on the previous.

```bash
# 1. Build story groups and train/val/test split assignments
python 01_data_engineering/scripts/build_splits.py

# 2. Add script-aware flag_junk, max_sim_train_other, story_disjoint columns
python 01_data_engineering/scripts/add_flags_and_similarity.py

# 3. Verify leakage invariants
python 01_data_engineering/scripts/check_leakage.py

# 4. Compute cross-split similarity report
python 01_data_engineering/scripts/cross_split_similarity.py

# 5. Compute group stats and write data_card_metrics.md/json
python 03_inference_evaluation/scripts/report_stats.py

# 6. Run shortcut baselines
python 02_model_training/scripts/shortcut_baselines.py

# 7. Generate this data card
python 03_inference_evaluation/scripts/generate_data_card.py
```
"""

    with open('01_data_engineering/DATA_CARD.md', 'w', encoding='utf-8') as f:
        f.write(card)
        
    print("Regenerated 01_data_engineering/DATA_CARD.md")

if __name__ == '__main__':
    import re
    main()
