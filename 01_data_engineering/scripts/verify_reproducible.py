import os
import sys
import subprocess
import hashlib
import pandas as pd
import shutil
import tempfile

def get_file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_cols_hash(df_path, cols):
    df = pd.read_csv(df_path)
    df_sorted = df.sort_values('row_id').reset_index(drop=True)
    return hashlib.sha256(df_sorted[cols].to_csv(index=False).encode('utf-8')).hexdigest()

def run_script(script, *args):
    cmd = [sys.executable, script] + list(args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    existing_splits = 'data/processed/splits_8class.csv'
    if not os.path.exists(existing_splits):
        print(f"ERROR: {existing_splits} not found.")
        return

    expected_full_sha = '8c05df8355c44611b05aa290c6644f822436734a74da278f65da1c91f370cf0b'
    current_full_sha = get_file_hash(existing_splits).lower()
    
    print(f"Current full CSV SHA-256:  {current_full_sha}")
    print(f"Expected full CSV SHA-256: {expected_full_sha}")
    
    key_cols = ['row_id', 'cluster', 'split', 'group_id']
    current_key_sha = get_cols_hash(existing_splits, key_cols)
    print(f"Current key cols SHA-256:  {current_key_sha}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_splits = os.path.join(temp_dir, 'splits_8class.csv')
        temp_report = os.path.join(temp_dir, 'split_report.md')
        
        # 1. build_splits.py
        run_script('01_data_engineering/scripts/build_splits.py', 
                   '--raw_path', 'data/raw/multitude_v3_clean.csv',
                   '--out_path', temp_splits,
                   '--report_path', temp_report)
                   
        # 2. add_flags_and_similarity.py
        run_script('01_data_engineering/scripts/add_flags_and_similarity.py', 
                   '--splits_path', temp_splits)
                   
        new_full_sha = get_file_hash(temp_splits).lower()
        new_key_sha = get_cols_hash(temp_splits, key_cols)
        
        print("\n--- RESULTS ---")
        print(f"Original full SHA-256: {current_full_sha}")
        print(f"Rebuilt full SHA-256:  {new_full_sha}")
        print(f"Original key cols SHA: {current_key_sha}")
        print(f"Rebuilt key cols SHA:  {new_key_sha}")
        
        if current_full_sha != new_full_sha or current_key_sha != new_key_sha:
            print("\nHashes differ! Analyzing nondeterminism...")
            orig_df = pd.read_csv(existing_splits).sort_values('row_id').reset_index(drop=True)
            new_df = pd.read_csv(temp_splits).sort_values('row_id').reset_index(drop=True)
            
            diff_split = (orig_df['split'] != new_df['split']).sum()
            diff_group = (orig_df['group_id'] != new_df['group_id']).sum()
            print(f"Rows with different split: {diff_split}")
            print(f"Rows with different group_id: {diff_group}")
            
            orig_counts = orig_df['split'].value_counts().to_dict()
            new_counts = new_df['split'].value_counts().to_dict()
            print(f"Original split counts: {orig_counts}")
            print(f"Rebuilt split counts:  {new_counts}")
        else:
            print("\nSuccess: Fully reproducible!")

if __name__ == '__main__':
    main()
