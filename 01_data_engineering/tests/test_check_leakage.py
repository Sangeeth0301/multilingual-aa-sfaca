import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.check_leakage import check_leakage

@pytest.fixture
def clean_mock_data(tmp_path):
    # Create raw
    raw = pd.DataFrame({
        'row_id': [0, 1, 2, 3],
        'text': ['Hello world', 'Hello world!', 'Bonjour', 'Hola'],
        'language': ['en', 'en', 'fr', 'es'], # fr is not in BASE_PAPER_18, wait fr is? No, base paper has 'pt', 'ro', 'es'. So let's use 'es'
        'multi_label': ['human', 'gpt3', 'human', 'human'],
        'split': ['train', 'train', 'test', 'test']
    })
    raw['language'] = ['en', 'en', 'es', 'es'] # Fix languages
    raw_path = tmp_path / 'raw.csv'
    raw.to_csv(raw_path, index=False)
    
    # Create splits
    splits = pd.DataFrame({
        'row_id': [0, 2, 3],
        'cluster': ['latin', 'latin', 'latin'],
        'language': ['en', 'es', 'es'],
        'multi_label': ['human', 'human', 'human'],
        'split': ['train', 'test', 'test'],
        'group_id': [1, -1, -1]
    })
    # Notice we drop row 1 (hash overlap with 0? wait 0 and 1 are train, if they overlap it's fine as long as not cross split, but we said no train/val/test overlap)
    splits_path = tmp_path / 'splits.csv'
    splits.to_csv(splits_path, index=False)
    
    return str(splits_path), str(raw_path)

def test_leakage_clean(clean_mock_data, capsys):
    splits_path, raw_path = clean_mock_data
    assert check_leakage(splits_path, raw_path) == True
    out, _ = capsys.readouterr()
    assert "SUCCESS" in out

def test_leakage_hash_overlap(clean_mock_data, capsys):
    splits_path, raw_path = clean_mock_data
    # Introduce hash overlap between train and val to avoid test set size mismatch
    splits = pd.read_csv(splits_path)
    # add a row to val that overlaps with train
    new_row = pd.DataFrame({'row_id': [1], 'cluster': ['latin'], 'language': ['en'], 'multi_label': ['human'], 'split': ['val'], 'group_id': [5]})
    splits = pd.concat([splits, new_row])
    splits.to_csv(splits_path, index=False)
    
    assert check_leakage(splits_path, raw_path) == False
    out, _ = capsys.readouterr()
    assert "FAIL: Train and Val have 1 overlapping hashes." in out

def test_leakage_group_overlap(clean_mock_data, capsys):
    splits_path, raw_path = clean_mock_data
    splits = pd.read_csv(splits_path)
    # Add a row to val with same group_id as train
    new_row = pd.DataFrame({'row_id': [9], 'cluster': ['latin'], 'language': ['en'], 'multi_label': ['human'], 'split': ['val'], 'group_id': [1]})
    splits = pd.concat([splits, new_row])
    splits.to_csv(splits_path, index=False)
    
    raw = pd.read_csv(raw_path)
    new_raw = pd.DataFrame({'row_id': [9], 'text': ['Completely different'], 'language': ['en'], 'multi_label': ['human'], 'split': ['train']})
    raw = pd.concat([raw, new_raw])
    raw.to_csv(raw_path, index=False)
    
    assert check_leakage(splits_path, raw_path) == False
    out, _ = capsys.readouterr()
    assert "FAIL: Found 1 groups spanning multiple splits." in out

def test_leakage_gemini(clean_mock_data, capsys):
    splits_path, raw_path = clean_mock_data
    splits = pd.read_csv(splits_path)
    splits.loc[0, 'multi_label'] = 'Gemini-1.5'
    splits.to_csv(splits_path, index=False)
    
    assert check_leakage(splits_path, raw_path) == False
    out, _ = capsys.readouterr()
    assert "FAIL: Found Gemini/Claude labels in splits." in out

def test_leakage_empty_hash_ok(clean_mock_data, capsys):
    splits_path, raw_path = clean_mock_data
    # Empty hashes shouldn't trigger the leak check
    splits = pd.read_csv(splits_path)
    new_row = pd.DataFrame({'row_id': [4], 'cluster': ['latin'], 'language': ['en'], 'multi_label': ['human'], 'split': ['val'], 'group_id': [9]})
    splits = pd.concat([splits, new_row])
    splits.to_csv(splits_path, index=False)
    
    raw = pd.read_csv(raw_path)
    # text that normalizes to empty
    new_raw = pd.DataFrame({'row_id': [4], 'text': ['... /// ...'], 'language': ['en'], 'multi_label': ['human'], 'split': ['val']})
    raw = pd.concat([raw, new_raw])
    # Also add an empty hash to train
    new_raw_2 = pd.DataFrame({'row_id': [5], 'text': ['-- --'], 'language': ['en'], 'multi_label': ['human'], 'split': ['train']})
    raw = pd.concat([raw, new_raw_2])
    raw.to_csv(raw_path, index=False)
    
    new_split_row = pd.DataFrame({'row_id': [5], 'cluster': ['latin'], 'language': ['en'], 'multi_label': ['human'], 'split': ['train'], 'group_id': [10]})
    splits = pd.concat([splits, new_split_row])
    splits.to_csv(splits_path, index=False)
    
    assert check_leakage(splits_path, raw_path) == True
