import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from src.clusters import get_cluster, BASE_PAPER_18_LANGUAGES

def main():
    csv_path = "data/raw/multitude_v3_clean.csv"
    df = pd.read_csv(csv_path)
    
    # Filter to base paper 18 languages
    df = df[df["language"].isin(BASE_PAPER_18_LANGUAGES)].reset_index(drop=True)
    df["cluster"] = df["language"].apply(get_cluster)
    
    counts = df["cluster"].value_counts()
    print("Cluster counts:")
    for cluster, count in counts.items():
        # Train split is 80% (according to train.py data loading)
        # But let's get the exact split from train.py logic:
        # train_df = filter_by_split(cluster_df, "train")
        train_df = df[(df["cluster"] == cluster) & (df["split"] == "train")]
        val_df = df[(df["cluster"] == cluster) & (df["split"] == "test")]
        print(f"  {cluster}: {count} total samples ({len(train_df)} train, {len(val_df)} test)")

if __name__ == "__main__":
    main()
