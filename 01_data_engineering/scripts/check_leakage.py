import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)
import pandas as pd
import hashlib
import string
import re
import os
import sys


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

def check_leakage(splits_path='data/processed/splits_8class.csv', raw_path='data/raw/multitude_v3_clean.csv'):
    if not os.path.exists(splits_path) or not os.path.exists(raw_path):
        print(f"Error: Required files not found.")
        return False
        
    splits = pd.read_csv(splits_path)
    raw = pd.read_csv(raw_path)
    
    # 1. No ca/ga/gd rows anywhere
    invalid_langs = set(splits['language']).intersection({'ca', 'ga', 'gd'})
    if invalid_langs:
        print(f"FAIL: Found invalid languages {invalid_langs}")
        return False
        
    # 2. No Gemini/Claude labels anywhere
    invalid_labels = splits[splits['multi_label'].astype(str).str.contains('Gemini|Claude', case=False, na=False)]
    if len(invalid_labels) > 0:
        print("FAIL: Found Gemini/Claude labels in splits.")
        return False
        
    # Check id column name mapping
    if 'Unnamed: 0' in raw.columns:
        raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        raw['row_id'] = raw.index
        
    # Merge text back into splits to compute hashes
    splits = splits.merge(raw[['row_id', 'text', 'split']], on='row_id', how='left', suffixes=('', '_orig'))
    splits['norm_hash'] = splits['text'].apply(lambda x: hash_text(normalize_text(x)))
    
    # 3. Test rows are EXACTLY the original test rows of the 18 languages
    # Original test rows of the 18 languages
    from clusters import BASE_PAPER_18_LANGUAGES
    orig_test = raw[(raw['split'] == 'test') & (raw['language'].isin(BASE_PAPER_18_LANGUAGES))]
    # But wait, original test also has Gemini/Claude? The instructions said "Drop Gemini/Claude labels".
    # And "Test rows are exactly the original test rows of the 18 languages". Does that mean keeping Gemini/Claude in test?
    # Actually, phase B said "Remove the template-based Gemini/Claude rows from every training and evaluation path."
    # So original test minus Gemini/Claude.
    orig_test = orig_test[~orig_test['multi_label'].astype(str).str.contains('Gemini|Claude', case=False, na=False)]
    
    current_test = splits[splits['split'] == 'test']
    
    if len(orig_test) != len(current_test):
        print(f"FAIL: Test set size mismatch. Expected {len(orig_test)}, got {len(current_test)}")
        return False
        
    if not set(orig_test['row_id']) == set(current_test['row_id']):
        print("FAIL: Test set row IDs do not perfectly match the expected original test rows.")
        return False

    # 4. No normalized-hash overlap among train/val/test (excluding empty hashes)
    empty_hash = hash_text("")
    train_hashes = set(splits[(splits['split'] == 'train') & (splits['norm_hash'] != empty_hash)]['norm_hash'])
    val_hashes = set(splits[(splits['split'] == 'val') & (splits['norm_hash'] != empty_hash)]['norm_hash'])
    test_hashes = set(splits[(splits['split'] == 'test') & (splits['norm_hash'] != empty_hash)]['norm_hash'])
    
    if train_hashes.intersection(val_hashes):
        print(f"FAIL: Train and Val have {len(train_hashes.intersection(val_hashes))} overlapping hashes.")
        return False
    if train_hashes.intersection(test_hashes):
        print(f"FAIL: Train and Test have {len(train_hashes.intersection(test_hashes))} overlapping hashes.")
        return False
    if val_hashes.intersection(test_hashes):
        print(f"FAIL: Val and Test have {len(val_hashes.intersection(test_hashes))} overlapping hashes.")
        return False
        
    # 5. No group_id in more than one split (ignore -1)
    # group_id -1 is allowed to be anywhere
    grouped = splits[splits['group_id'] != -1].groupby('group_id')['split'].nunique()
    invalid_groups = grouped[grouped > 1]
    if len(invalid_groups) > 0:
        print(f"FAIL: Found {len(invalid_groups)} groups spanning multiple splits.")
        return False

    print("SUCCESS: All leakage checks passed.")
    return True

if __name__ == '__main__':
    if not check_leakage():
        sys.exit(1)
    sys.exit(0)
