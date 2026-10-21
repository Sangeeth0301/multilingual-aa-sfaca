"""
01_data_engineering/scripts/group_structure.py
Reports per-(language, outlet, generator) machine/human text ratios in train rows.
Output: results/group_structure.md
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)
import argparse
import sys
import os
import pandas as pd
import numpy as np

from clusters import BASE_PAPER_18_LANGUAGES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='data/raw/multitude_v3_clean.csv')
    parser.add_argument('--splits_path', default='data/processed/splits_8class.csv')
    parser.add_argument('--out_md', default='results/group_structure.md')
    args = parser.parse_args()

    if not os.path.exists(args.raw_path):
        print(f"ERROR: {args.raw_path} not found. Aborting.")
        sys.exit(1)
    if not os.path.exists(args.splits_path):
        print(f"ERROR: {args.splits_path} not found. Aborting.")
        sys.exit(1)

    raw = pd.read_csv(args.raw_path)
    if 'Unnamed: 0' in raw.columns:
        raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        raw['row_id'] = raw.index

    splits = pd.read_csv(args.splits_path)

    # Join to get split assignments
    df = splits.merge(raw[['row_id', 'source']], on='row_id', how='left')
    df['source'] = df['source'].astype(str).str.replace('MULTITuDE_', '')
    # Only train rows for ratio analysis (train+val as assigned by our pipeline)
    tv = df[df['split'].isin(['train', 'val'])].copy()
    tv = tv[tv['language'].isin(BASE_PAPER_18_LANGUAGES)].copy()

    generators = sorted([g for g in tv['multi_label'].unique() if g != 'human'])
    humans = tv[tv['multi_label'] == 'human'].groupby(['language', 'source']).size().rename('n_human')

    rows = []
    for gen in generators:
        machines = tv[tv['multi_label'] == gen].groupby(['language', 'source']).size().rename('n_machine')
        combined = pd.concat([humans, machines], axis=1)
        combined = combined[combined['n_human'].notna() & combined['n_machine'].notna()]
        if len(combined) == 0:
            continue
        ratio = combined['n_machine'] / combined['n_human']
        n_buckets = len(ratio)
        in_range = ((ratio >= 0.9) & (ratio <= 1.1)).sum()
        for (lang, src), r in ratio.items():
            rows.append({
                'generator': gen,
                'language': lang,
                'source': src,
                'n_human': int(combined.loc[(lang, src), 'n_human']),
                'n_machine': int(combined.loc[(lang, src), 'n_machine']),
                'ratio': round(r, 4),
            })

    ratio_df = pd.DataFrame(rows)

    lines = ["# Group Structure: Machine/Human Ratios per (Language, Outlet, Generator)\n"]
    lines.append("Computed from `data/processed/splits_8class.csv` (train+val rows).\n")
    lines.append("## Summary per Generator\n")
    lines.append("| Generator | Buckets | Share in [0.9,1.1] | Min ratio | Max ratio | Mean ratio |")
    lines.append("|-----------|---------|-------------------|-----------|-----------|------------|")

    for gen in generators:
        sub = ratio_df[ratio_df['generator'] == gen]
        if len(sub) == 0:
            continue
        n = len(sub)
        in_r = ((sub['ratio'] >= 0.9) & (sub['ratio'] <= 1.1)).sum()
        lines.append(
            f"| {gen} | {n} | {in_r/n*100:.1f}% | {sub['ratio'].min():.3f} | {sub['ratio'].max():.3f} | {sub['ratio'].mean():.3f} |"
        )

    lines.append("\n## Worst Buckets (ratio < 0.9 or > 1.1)\n")
    bad = ratio_df[(ratio_df['ratio'] < 0.9) | (ratio_df['ratio'] > 1.1)].copy()
    bad = bad.sort_values('ratio')
    if len(bad) == 0:
        lines.append("_No buckets outside [0.9, 1.1]._\n")
    else:
        lines.append("| Generator | Language | Source | n_human | n_machine | ratio |")
        lines.append("|-----------|----------|--------|---------|-----------|-------|")
        for _, r in bad.iterrows():
            lines.append(
                f"| {r['generator']} | {r['language']} | {r['source']} | {r['n_human']} | {r['n_machine']} | {r['ratio']:.3f} |"
            )

    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Written to {args.out_md}")
    print(f"Total (lang,outlet,generator) buckets: {len(ratio_df)}")
    if len(ratio_df) > 0:
        in_r = ((ratio_df['ratio'] >= 0.9) & (ratio_df['ratio'] <= 1.1)).sum()
        print(f"Share in [0.9,1.1]: {in_r/len(ratio_df)*100:.1f}%")
        print(f"Global min ratio: {ratio_df['ratio'].min():.3f}")
        print(f"Global max ratio: {ratio_df['ratio'].max():.3f}")


if __name__ == '__main__':
    main()
