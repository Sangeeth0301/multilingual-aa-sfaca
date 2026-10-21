import argparse
import pandas as pd
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='data/raw/multitude_v3_clean.csv')
    parser.add_argument('--splits_path', default='data/processed/splits_8class.csv')
    parser.add_argument('--out_md', default='results/data_card_metrics.md')
    parser.add_argument('--out_json', default='results/data_card_metrics.json')
    args = parser.parse_args()

    splits = pd.read_csv(args.splits_path)
    raw = pd.read_csv(args.raw_path)
    if 'Unnamed: 0' in raw.columns:
        raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        raw['row_id'] = raw.index

    df = splits.merge(raw[['row_id', 'source']], on='row_id', how='left')
    grouped_df = df[df['group_id'] != -1]

    num_groups = grouped_df['group_id'].nunique()
    num_human = grouped_df[grouped_df['multi_label'] == 'human']['row_id'].nunique()

    group_counts = grouped_df.groupby('group_id')['multi_label'].value_counts().unstack(fill_value=0)
    if 'human' not in group_counts.columns:
        group_counts['human'] = 0
    group_counts['machine'] = group_counts.sum(axis=1) - group_counts['human']
    humans_no_machine = (group_counts['human'] > 0) & (group_counts['machine'] == 0)
    humans_no_machine_count = int(humans_no_machine.sum())
    humans_no_machine_pct = (humans_no_machine_count / num_human) * 100 if num_human > 0 else 0

    group_sizes = grouped_df.groupby('group_id').size()
    sz_min = int(group_sizes.min())
    sz_median = float(group_sizes.median())
    sz_p90 = float(np.percentile(group_sizes, 90))
    sz_max = int(group_sizes.max())

    top_groups = []
    largest = group_sizes.nlargest(10)
    for g_id, size in largest.items():
        g_df = grouped_df[grouped_df['group_id'] == g_id]
        lang = g_df['language'].iloc[0]
        outlet = g_df['source'].iloc[0]
        split_dist = g_df['split'].value_counts().to_dict()
        top_groups.append({'group_id': int(g_id), 'size': int(size), 'language': lang, 'outlet': outlet, 'splits': split_dist})

    # Junk counts
    junk_counts = df[df['flag_junk']].groupby(['cluster', 'split', 'multi_label']).size().reset_index(name='count')
    junk_counts_list = junk_counts.to_dict('records')

    # Story disjointness
    disjoint = {}
    for split in ['val', 'test']:
        disjoint[split] = {}
        sub = df[df['split'] == split]
        for lang in sorted(sub['language'].unique()):
            lang_sub = sub[sub['language'] == lang]
            disjoint_pct = lang_sub['story_disjoint'].mean() * 100
            disjoint[split][lang] = float(disjoint_pct)

    metrics = {
        'groups': {
            'num_groups': num_groups,
            'num_human_anchors': num_human,
            'humans_no_machine_count': humans_no_machine_count,
            'humans_no_machine_pct': humans_no_machine_pct,
            'size_min': sz_min,
            'size_median': sz_median,
            'size_p90': sz_p90,
            'size_max': sz_max,
            'top_10_largest': top_groups
        },
        'junk_counts': junk_counts_list,
        'story_disjointness': disjoint
    }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write("# Data Card Metrics\n\n")
        f.write("## Groups (Train + Val)\n")
        f.write(f"- Test rows have group_id -1 and are not counted here.\n")
        f.write(f"- Number of human anchors: {num_human}\n")
        f.write(f"- Humans with no attached machine text: {humans_no_machine_count} ({humans_no_machine_pct:.1f}%)\n")
        f.write(f"- Group sizes: min={sz_min}, median={sz_median}, p90={sz_p90}, max={sz_max}\n\n")
        f.write("### Top 10 Largest Groups\n")
        for g in top_groups:
            f.write(f"- Group {g['group_id']}: size {g['size']}, lang {g['language']}, outlet {g['outlet']}, splits {g['splits']}\n")
        
        f.write("\n## Junk Counts (New Script-Aware)\n")
        for row in junk_counts_list:
            f.write(f"- {row['cluster']} / {row['split']} / {row['multi_label']}: {row['count']}\n")
        
        f.write("\n## Story Disjointness\n")
        for split in ['val', 'test']:
            f.write(f"\n### {split.upper()}\n")
            for lang, pct in disjoint[split].items():
                f.write(f"- {lang}: {pct:.1f}% disjoint\n")
    
    print(f"Metrics written to {args.out_md} and {args.out_json}")

if __name__ == '__main__':
    main()
