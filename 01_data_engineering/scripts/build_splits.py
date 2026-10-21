"""
01_data_engineering/scripts/build_splits.py
Builds train/val/test splits for MULTITuDE with one-to-one story grouping.

Strategy
--------
Per (language, outlet, generator) bucket: maximise total TF-IDF cosine by
solving a 1:1 assignment (scipy.optimize.linear_sum_assignment).
Each human text receives at most one machine text per generator.
Leftover machine texts (|machine| > |human|) are greedily attached to their
best-available human.  Leftover human texts (no machine text in a bucket) form
singleton groups.

Test rows are kept exactly as they are (group_id = -1).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)
import argparse
import hashlib
import json
import os
import re
import regex
import string
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from clusters import get_cluster, BASE_PAPER_18_LANGUAGES, LANG_TO_CLUSTER


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).strip()


def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()


def detect_dominant_script(text):
    counts = {
        'latin':    len(regex.findall(r'\p{IsLatin}', text)),
        'cyrillic': len(regex.findall(r'\p{IsCyrillic}', text)),
        'greek':    len(regex.findall(r'\p{IsGreek}', text)),
        'arabic':   len(regex.findall(r'\p{IsArabic}', text)),
        'hanzi':    len(regex.findall(r'\p{IsHan}', text)),
    }
    total = sum(counts.values())
    return 'unknown' if total == 0 else max(counts, key=counts.get)


# ── one-to-one assignment for one (language, outlet) bucket ──────────────────

def assign_one_to_one(humans_df, machines_per_gen, vectorizer):
    """
    Returns dict  human_row_id -> list[machine_row_id]
    across all generators in this (language, outlet) bucket.

    For each generator independently:
      - solve linear_sum_assignment to maximise cosine similarity
      - any leftover machine texts (|machine|>|human|) attach to best human
    """
    all_texts = list(humans_df['text'].fillna(''))
    human_ids  = list(humans_df['row_id'])
    n_h = len(human_ids)

    # build mapping human_row_id -> list of assigned machine row_ids
    assignments = {h: [] for h in human_ids}

    for gen, mach_df in machines_per_gen.items():
        if len(mach_df) == 0:
            continue

        mach_texts = list(mach_df['text'].fillna(''))
        mach_ids   = list(mach_df['row_id'])
        n_m = len(mach_ids)

        all_t = all_texts + mach_texts
        try:
            tfidf = vectorizer.fit_transform(all_t)
        except ValueError:
            # fallback: attach each machine to the first human
            for mid in mach_ids:
                assignments[human_ids[0]].append(mid)
            continue

        h_vecs = tfidf[:n_h]
        m_vecs = tfidf[n_h:]
        sim    = cosine_similarity(m_vecs, h_vecs)  # (n_m, n_h)

        if n_m <= n_h:
            # more-or-equal humans: 1:1, each machine to unique human
            row_ind, col_ind = linear_sum_assignment(-sim)
            for r, c in zip(row_ind, col_ind):
                assignments[human_ids[c]].append(mach_ids[r])
        else:
            # more machines than humans: 1:1 for first n_h, then greedy for rest
            row_ind, col_ind = linear_sum_assignment(-sim[:n_h, :])   # only top n_h rows
            matched_machines = set()
            for r, c in zip(row_ind, col_ind):
                assignments[human_ids[c]].append(mach_ids[r])
                matched_machines.add(r)
            # leftover machines: greedy best human
            for r in range(n_m):
                if r not in matched_machines:
                    best_h = int(np.argmax(sim[r]))
                    assignments[human_ids[best_h]].append(mach_ids[r])

    return assignments


# ── main ─────────────────────────────────────────────────────────────────────

def build_splits():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path',    default='data/raw/multitude_v3_clean.csv')
    parser.add_argument('--out_path',    default='data/processed/splits_8class.csv')
    parser.add_argument('--report_path', default='results/split_report.md')
    args = parser.parse_args()

    raw_path    = args.raw_path
    out_path    = args.out_path
    report_path = args.report_path

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if os.path.dirname(report_path):
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found.  Aborting.")
        sys.exit(1)

    raw_sha256 = get_file_hash(raw_path)
    print("Loading data...")
    df = pd.read_csv(raw_path)

    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        df['row_id'] = df.index

    df['text'] = df['text'].fillna('')
    df['norm_hash'] = df['text'].apply(lambda x: hash_text(normalize_text(x)))

    # ── flags ────────────────────────────────────────────────────────────────
    print("Calculating flags...")
    df['flag_junk'] = df['text'].apply(lambda x: len(regex.findall(r'\p{L}', x)) < 20)

    df['cluster'] = df['language'].apply(
        lambda x: get_cluster(x) if x in LANG_TO_CLUSTER else 'unknown')
    df['dominant_script'] = df['text'].apply(detect_dominant_script)
    df['flag_script_mismatch'] = df['dominant_script'] != df['cluster']

    prompt_patterns = (
        r'(?i)\b(task write|task translate|leave out the|return just the'
        r'|you are a|your article in|a news article in|the article in'
        r'|text of the|here is a|title:)\b'
    )
    prompt_regex = re.compile(prompt_patterns)
    df['flag_prompt_echo'] = df['text'].apply(
        lambda x: bool(prompt_regex.search(x[:200])))

    print("Tokenizing for lengths...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    df['length_tokens'] = df['text'].apply(
        lambda x: len(tokenizer.encode(
            x, add_special_tokens=True, max_length=512, truncation=True)))

    empty_hash = hash_text("")

    # ── language filter ───────────────────────────────────────────────────────
    ca_ga_gd = df[df['language'].isin(['ca', 'ga', 'gd'])]
    os.makedirs('results', exist_ok=True)
    ca_ga_gd[['row_id']].to_csv('results/ca_ga_gd_ids.csv', index=False)
    df = df[df['language'].isin(BASE_PAPER_18_LANGUAGES)].copy()

    # drop Gemini/Claude
    df = df[~df['multi_label'].astype(str).str.contains('Gemini|Claude',
                                                          case=False, na=False)]

    test_df  = df[df['split'] == 'test'].copy()
    train_df = df[df['split'] == 'train'].copy()

    # purge test hashes from train (excluding empty hash)
    test_hashes = set(test_df[test_df['norm_hash'] != empty_hash]['norm_hash'])
    train_df = train_df[
        ~((train_df['norm_hash'].isin(test_hashes)) &
          (train_df['norm_hash'] != empty_hash))].copy()

    # within-train dedup
    train_df = train_df.drop_duplicates(subset=['norm_hash'], keep='first').copy()

    train_df['outlet'] = train_df['source'].astype(str).str.replace('MULTITuDE_', '')

    # ── one-to-one grouping ───────────────────────────────────────────────────
    print("Building one-to-one story groups (linear_sum_assignment)...")
    np.random.seed(42)

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                 max_features=5000)

    train_df = train_df.reset_index(drop=True)
    group_id_col = np.full(len(train_df), -1, dtype=np.int64)
    gid_counter = 0

    leftover_counts = []   # (lang, outlet, gen, n_leftover)

    for (lang, outlet), bucket_df in train_df.groupby(['language', 'outlet']):
        humans_df  = bucket_df[bucket_df['label'] == 0].copy()
        machines_df = bucket_df[bucket_df['label'] == 1].copy()

        if len(humans_df) == 0:
            # no humans: each machine is its own singleton group
            for i in bucket_df.index:
                group_id_col[i] = gid_counter
                gid_counter += 1
            continue

        if len(machines_df) == 0:
            # no machines: each human is its own singleton group
            for i in bucket_df.index:
                group_id_col[i] = gid_counter
                gid_counter += 1
            continue

        machines_per_gen = {
            gen: machines_df[machines_df['multi_label'] == gen]
            for gen in machines_df['multi_label'].unique()
        }

        try:
            assignments = assign_one_to_one(humans_df, machines_per_gen, vectorizer)
        except Exception as e:
            print(f'Error in one-to-one: {e}')
            # fallback: simple greedy as before
            for i in bucket_df.index:
                group_id_col[i] = gid_counter
                gid_counter += 1
            continue

        # count leftovers per generator
        for gen, mach_df in machines_per_gen.items():
            n_h = len(humans_df)
            n_m = len(mach_df)
            n_leftover = max(0, n_m - n_h)
            if n_leftover > 0:
                leftover_counts.append((lang, outlet, gen, n_leftover))

        # assign group ids: human + its machines share a group id
        # build reverse map: row_id -> group_id
        row_id_to_gid = {}
        for h_id, m_ids in assignments.items():
            row_id_to_gid[h_id] = gid_counter
            for m_id in m_ids:
                row_id_to_gid[m_id] = gid_counter
            gid_counter += 1

        # any humans not in assignments (shouldn't happen) get singleton group
        for _, row in humans_df.iterrows():
            if row['row_id'] not in row_id_to_gid:
                row_id_to_gid[row['row_id']] = gid_counter
                gid_counter += 1

        # write back
        row_id_series = train_df.loc[bucket_df.index, 'row_id']
        for i, rid in zip(bucket_df.index, row_id_series):
            if rid in row_id_to_gid:
                group_id_col[i] = row_id_to_gid[rid]
            else:
                group_id_col[i] = gid_counter
                gid_counter += 1

    train_df['group_id'] = group_id_col

    # ── group diagnostics ─────────────────────────────────────────────────────
    print("Extracting grouping diagnostics...")
    group_sizes = train_df.groupby('group_id').size()
    total_groups    = len(group_sizes)
    singletons      = int((group_sizes == 1).sum())
    groups_1_human  = int(
        (train_df[train_df['label'] == 0].groupby('group_id').size() == 1).sum())
    groups_gt8      = int((group_sizes > 8).sum())

    # ── val split ─────────────────────────────────────────────────────────────
    print("StratifiedGroupKFold Validation Splitting...")
    train_df['final_split'] = 'train'
    val_indices = []

    for cluster in train_df['cluster'].unique():
        c_df = train_df[train_df['cluster'] == cluster]
        if len(c_df) < 10:
            continue
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        try:
            train_idx, val_idx = next(
                sgkf.split(c_df, c_df['multi_label'], c_df['group_id']))
            val_indices.extend(c_df.iloc[val_idx].index)
        except ValueError:
            pass

    train_df.loc[val_indices, 'final_split'] = 'val'
    test_df['group_id'] = -1
    test_df['final_split'] = 'test'

    final_cols = [
        'row_id', 'cluster', 'language', 'multi_label', 'final_split',
        'group_id', 'flag_junk', 'flag_script_mismatch', 'flag_prompt_echo',
        'length_tokens',
    ]
    final_df = pd.concat([train_df[final_cols], test_df[final_cols]])
    final_df = final_df.rename(columns={'final_split': 'split'})

    final_df.to_csv(out_path, index=False)
    out_sha256 = get_file_hash(out_path)

    # ── report ────────────────────────────────────────────────────────────────
    print("Writing report...")
    report = ["# Split Design Report\n"]
    report.append(f"- **Raw CSV SHA-256**: {raw_sha256}")
    report.append(f"- **Splits CSV SHA-256**: {out_sha256}")
    report.append(f"- **Grouping strategy**: one-to-one per (language, outlet, generator) via `linear_sum_assignment`")

    report.append("\n## Story Grouping Diagnostics")
    report.append(f"- **Total groups**: {total_groups}")
    report.append(f"- **Singletons**: {singletons} ({singletons/total_groups:.2%})")
    report.append(f"- **Groups with exactly 1 human**: {groups_1_human}")
    report.append(f"- **Groups > 8 rows**: {groups_gt8}")
    report.append(f"- **Group sizes** — min: {group_sizes.min()}, median: {group_sizes.median():.1f}, "
                  f"p90: {np.percentile(group_sizes, 90):.1f}, max: {group_sizes.max()}")
    report.append(f"\n### Val rows in groups > 8, per cluster")
    final_tv = final_df[final_df['split'].isin(['train','val'])]
    for cl in sorted(final_tv['cluster'].unique()):
        big_g  = group_sizes[group_sizes > 8].index
        cl_val = final_df[(final_df['cluster']==cl) & (final_df['split']=='val')]
        n_val  = len(cl_val)
        n_big  = int(cl_val['group_id'].isin(big_g).sum())
        report.append(f"- {cl}: {n_big}/{n_val} val rows in groups > 8 ({n_big/n_val*100:.1f}% )" if n_val > 0 else f"- {cl}: 0 val rows")

    report.append("\n### Leftover machine texts (|machine|>|human| in a bucket)")
    if leftover_counts:
        report.append("| language | outlet | generator | n_leftover |")
        report.append("|----------|--------|-----------|------------|")
        for lang, outlet, gen, n in sorted(leftover_counts, key=lambda x: -x[3]):
            report.append(f"| {lang} | {outlet} | {gen} | {n} |")
    else:
        report.append("_None — every bucket has |machine| ≤ |human| for every generator._")

    report.append("\n## Group Size Histogram")
    size_counts = group_sizes.value_counts().sort_index()
    for s, c in size_counts.items():
        if s <= 15 or s % 5 == 0:
            report.append(f"- Size {s}: {c} groups")

    report.append("\n## Class Balance (Counts & Min/Max Ratio)")
    report.append("| Cluster | Split | Max/Min Ratio | Counts |")
    report.append("|---------|-------|---------------|--------|")
    for cluster in sorted(final_df['cluster'].unique()):
        for split in ['train', 'val', 'test']:
            c_df = final_df[(final_df['cluster'] == cluster) &
                            (final_df['split'] == split)]
            if len(c_df) == 0:
                continue
            counts = c_df['multi_label'].value_counts()
            ratio  = counts.max() / counts.min() if counts.min() > 0 else float('inf')
            counts_str = ', '.join(f"{k}: {v}" for k, v in counts.items())
            report.append(f"| {cluster} | {split} | {ratio:.2f} | {counts_str} |")

    report.append("\n### Prompt patterns flagged")
    report.append(f"`{prompt_patterns}`")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"Done. Splits written to {out_path}, report to {report_path}")


if __name__ == '__main__':
    build_splits()
