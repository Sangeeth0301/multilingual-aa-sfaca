import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)
import pandas as pd
import json
import os
import re
import string
import hashlib
from collections import defaultdict
import numpy as np

import sys

from transformers import AutoTokenizer
from clusters import get_cluster

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text.strip()

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def inspect_data():
    csv_path = 'data/raw/multitude_v3_clean.csv'
    report_md = []
    report_json = {}
    report_md.append("# Data Inspection Report\n")
    
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' does not exist. Please upload the data file before running this script.")
        sys.exit(1)
        
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # 1. Columns, dtypes, row count, counts per ...
    report_md.append("## 1. Dataset Overview\n")
    report_md.append(f"- **Row count**: {len(df)}")
    report_md.append(f"- **Columns**: {list(df.columns)}")
    report_md.append("- **Dtypes**:")
    for col, dtype in df.dtypes.items():
        report_md.append(f"  - {col}: {dtype}")
    
    report_json["row_count"] = len(df)
    report_json["columns"] = list(df.columns)
        
    # Assign clusters using 01_data_engineering/src/clusters.py
    if 'language' in df.columns:
        try:
            from clusters import LANG_TO_CLUSTER
            df['cluster'] = df['language'].apply(lambda x: get_cluster(x) if x in LANG_TO_CLUSTER else 'unknown')
        except ImportError:
            df['cluster'] = 'unknown'
    else:
        report_md.append("WARNING: 'language' column not found for clustering.")

    split_counts = df['split'].value_counts().to_dict() if 'split' in df.columns else {}
    lang_counts = df['language'].value_counts().to_dict() if 'language' in df.columns else {}
    label_counts = df['multi_label'].value_counts().to_dict() if 'multi_label' in df.columns else {}
    cluster_counts = df['cluster'].value_counts().to_dict() if 'cluster' in df.columns else {}
    
    report_md.append("\n### Counts")
    report_md.append("**Per Split:**\n" + "\n".join([f"- {k}: {v}" for k,v in split_counts.items()]))
    report_md.append("**Per Cluster:**\n" + "\n".join([f"- {k}: {v}" for k,v in cluster_counts.items()]))
    report_md.append("**Per Language:**\n" + "\n".join([f"- {k}: {v}" for k,v in lang_counts.items()]))
    report_md.append("**Per Label (multi_label):**\n" + "\n".join([f"- {k}: {v}" for k,v in label_counts.items()]))
    
    # Cross-tab
    report_md.append("\n**Cross-tab of cluster x label x split:**\n")
    if all(c in df.columns for c in ['cluster', 'multi_label', 'split']):
        crosstab = df.groupby(['cluster', 'multi_label', 'split']).size().reset_index(name='count')
        
        md_table = "| Cluster | Label | Split | Count |\n|---|---|---|---|\n"
        for _, row in crosstab.iterrows():
            md_table += f"| {row['cluster']} | {row['multi_label']} | {row['split']} | {row['count']} |\n"
        report_md.append(md_table)
        
        ct_dict = []
        for _, row in crosstab.iterrows():
            ct_dict.append({
                "cluster": row['cluster'],
                "label": row['multi_label'],
                "split": row['split'],
                "count": row['count']
            })
        report_json["crosstab"] = ct_dict

    # 2. Text length stats
    report_md.append("\n## 2. Text Length Statistics\n")
    print("Computing lengths...")
    df['char_length'] = df['text'].astype(str).apply(len)
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    print("Tokenizing...")
    df['tokens'] = df['text'].astype(str).apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True)))
    
    df['exceeds_128'] = df['tokens'] > 128
    
    report_md.append("| Cluster | Mean Chars | Mean Tokens | % > 128 tokens |")
    report_md.append("|---|---|---|---|")
    
    len_stats = {}
    for cluster in df['cluster'].unique():
        c_df = df[df['cluster'] == cluster]
        mean_char = c_df['char_length'].mean()
        mean_tok = c_df['tokens'].mean()
        pct_exceed = (c_df['exceeds_128'].sum() / len(c_df)) * 100
        report_md.append(f"| {cluster} | {mean_char:.1f} | {mean_tok:.1f} | {pct_exceed:.1f}% |")
        len_stats[cluster] = {
            "mean_char": float(mean_char),
            "mean_tok": float(mean_tok),
            "pct_exceed_128": float(pct_exceed)
        }
    report_json["length_stats"] = len_stats

    # 3. Source column
    report_md.append("\n## 3. Source Column Analysis\n")
    if 'source' in df.columns:
        source_counts = df['source'].value_counts().to_dict()
        report_md.append("**Unique values and counts:**\n" + "\n".join([f"- {k}: {v}" for k,v in source_counts.items()]))
        report_json["source_counts"] = source_counts
    else:
        report_md.append("No 'source' column found.")
    
    has_article_id = False
    for col in df.columns:
        if 'id' in col.lower() or 'article' in col.lower() or 'headline' in col.lower() or 'title' in col.lower():
            has_article_id = True
            report_md.append(f"\nPotential identifier column found: `{col}`.")
            break
    
    if not has_article_id:
        report_md.append("\n**Is there ANY column that identifies the original article or headline?**\n**NO.** There is no explicit identifier column.")

    # 4. Duplicates
    report_md.append("\n## 4. Duplicate Analysis\n")
    print("Finding duplicates...")
    df['norm_hash'] = df['text'].apply(lambda x: hash_text(normalize_text(x)))
    
    exact_dupes = df.duplicated(subset=['text'], keep=False).sum()
    near_dupes = df.duplicated(subset=['norm_hash'], keep=False).sum()
    
    train_hashes = set(df[df['split'] == 'train']['norm_hash'])
    test_hashes = set(df[df['split'] == 'test']['norm_hash'])
    train_test_overlap = len(train_hashes.intersection(test_hashes))
    overlap_rows_in_test = len(df[(df['split'] == 'test') & (df['norm_hash'].isin(train_hashes))])
    
    report_md.append(f"- Exact duplicate rows: {exact_dupes}")
    report_md.append(f"- Near-duplicate rows (normalized): {near_dupes}")
    report_md.append(f"- Test texts that also occur in train (leakage): {overlap_rows_in_test} rows ({train_test_overlap} unique hashes)")
    
    report_json["duplicates"] = {
        "exact_dupes": int(exact_dupes),
        "near_dupes": int(near_dupes),
        "test_in_train_overlap": int(overlap_rows_in_test)
    }

    # 5. Gemini/Claude rows
    report_md.append("\n## 5. Gemini/Claude Rows\n")
    gem_claude_rows = 0
    if 'multi_label' in df.columns:
        gem_claude_rows += df['multi_label'].astype(str).str.contains('Gemini|Claude', case=False, na=False).sum()
    if 'source' in df.columns:
        gem_claude_rows += df['source'].astype(str).str.contains('augmented|gemini|claude', case=False, na=False).sum()
        
    if gem_claude_rows > 0:
        report_md.append(f"**Found {gem_claude_rows} rows** containing Gemini/Claude references in label or source. The clean file should NOT contain them!")
    else:
        report_md.append("No Gemini/Claude rows found in label or source. Verified clean.")
    report_json["gemini_claude_rows"] = int(gem_claude_rows)

    # 6. Class Balance and KAGGLE_RUN_FACTS mismatch
    report_md.append("\n## 6. Class Balance and KAGGLE_RUN_FACTS Comparison\n")
    
    kaggle_facts = {
        'latin': {'train': 95431, 'test': 28631},
        'cyrillic': {'train': 23838, 'test': 7153},
        'greek': {'train': 7944, 'test': 2384},
        'arabic': {'train': 7975, 'test': 2392},
        'hanzi': {'train': 7926, 'test': 2383}
    }
    
    report_md.append("| Cluster | Split | Found | Kaggle Fact | Mismatch |")
    report_md.append("|---|---|---|---|---|")
    
    for cluster in ['latin', 'cyrillic', 'greek', 'arabic', 'hanzi']:
        for split in ['train', 'test']:
            found = len(df[(df['cluster'] == cluster) & (df['split'] == split)])
            fact = kaggle_facts[cluster][split]
            diff = found - fact
            mismatch = str(diff) if diff != 0 else "Match"
            report_md.append(f"| {cluster} | {split} | {found} | {fact} | {mismatch} |")

    # 7. 01_data_engineering/src/data_loader.py filters
    report_md.append("\n## 7. `01_data_engineering/src/data_loader.py` Filters\n")
    report_md.append("Looking at `01_data_engineering/src/data_loader.py`, the following filters apply when `load_multitude_csv(path, restrict_to_base_paper_18=True)` is called:")
    report_md.append("1. **Language Filter**: Drops 3 languages (`ca`, `ga`, `gd`) because they are not part of the base paper's 18 core languages.")
    report_md.append("2. **Label Verification**: Maps `multi_label` using `LABEL2ID`. Any row with an unrecognized `multi_label` raises a `ValueError` rather than silently dropping or mislabeling it.")

    # Split Design Proposal
    report_md.append("\n## 8. Split Design Proposal\n")
    proposal = """
### Grouping & Leakage
Since there is **no explicit original article/headline ID** in this dataset (`source` just identifies the dataset origin like 'multitude'), we cannot group strictly by article ID. 
To carve a validation set from `train` without leakage, we must group by **`norm_hash` (normalized text hash)**. 
**Limits:** Grouping by `norm_hash` only prevents identical or nearly-identical texts from crossing the train/val split. If a human article was rewritten by two different generators, they will have different hashes and might end up in different splits. This is an unavoidable limitation of not having article IDs. We must strictly ensure that `norm_hash` intersection between train and val is zero. Furthermore, any train texts that share a `norm_hash` with the fixed `test` set must be completely purged to eliminate test leakage.

### Duplicates
- **Test-in-Train Leakage**: All train rows that have a `norm_hash` present in `test` MUST BE DROPPED.
- **Within-Train Duplicates**: Retaining near-duplicates (e.g. same text generated multiple times or slight variations) within the same split (train or val) is generally safe, but deduplicating exact/near exact texts in train can prevent overfitting on common templates. We propose removing all within-train near-duplicates (`keep='first'`).

### Class Weighting
Class balance varies significantly per cluster (e.g., human vs AI classes, and between different generators). 
Because the loss is calculated over imbalanced classes, **Class Weighting** inside the CrossEntropyLoss is highly recommended, or at least a balanced sampler during training, to prevent the majority classes (often Human or a specific generator) from dominating the gradients.
"""
    report_md.append(proposal)

    os.makedirs('results', exist_ok=True)
    with open('results/data_inspection.json', 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
    
    with open('results/data_inspection.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report_md))
    
    print("Done. Saved to results/data_inspection.md and .json")

if __name__ == '__main__':
    inspect_data()
