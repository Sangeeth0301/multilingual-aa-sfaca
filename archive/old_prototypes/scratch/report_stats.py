import pandas as pd
import numpy as np

splits = pd.read_csv('data/processed/splits_8class.csv')
raw = pd.read_csv('data/raw/multitude_v3_clean.csv')

if 'Unnamed: 0' in raw.columns:
    raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
else:
    raw['row_id'] = raw.index

df = splits.merge(raw[['row_id', 'source']], on='row_id', how='left')

# Number of groups
num_groups = df['group_id'].nunique()
print(f"Number of groups: {num_groups}")

# Number of human anchors (assuming one human anchor per group, or counting human rows)
num_human = df[df['multi_label'] == 'human']['row_id'].nunique()
print(f"Number of human anchors: {num_human}")

# Humans with no attached machine text
# Group by group_id, count humans and machines
group_counts = df.groupby('group_id')['multi_label'].value_counts().unstack(fill_value=0)
group_counts['machine'] = group_counts.sum(axis=1) - group_counts.get('human', 0)
humans_no_machine = (group_counts.get('human', 0) > 0) & (group_counts['machine'] == 0)
print(f"Humans with no attached machine text: {humans_no_machine.sum()}")

# Group sizes
group_sizes = df.groupby('group_id').size()
print(f"Group size min: {group_sizes.min()}")
print(f"Group size median: {group_sizes.median()}")
print(f"Group size p90: {np.percentile(group_sizes, 90)}")
print(f"Group size max: {group_sizes.max()}")

# Top 10 largest groups
print("Top 10 largest groups:")
largest = group_sizes.nlargest(10)
for g_id, size in largest.items():
    g_df = df[df['group_id'] == g_id]
    lang = g_df['language'].iloc[0]
    outlet = g_df['source'].iloc[0]
    print(f"- Group {g_id}: size {size}, {lang}, {outlet}")
