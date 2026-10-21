import pandas as pd
import numpy as np
import os
import sys
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from scipy.sparse import hstack, csr_matrix
import re

def shortcut_baselines():
    splits_path = 'data/processed/splits_8class.csv'
    raw_path = 'data/raw/multitude_v3_clean.csv'
    out_path = 'results/shortcut_baselines.md'
    
    if not os.path.exists(splits_path):
        print("Splits not found")
        sys.exit(1)
        
    print("Loading data...")
    splits = pd.read_csv(splits_path)
    raw = pd.read_csv(raw_path)
    if 'Unnamed: 0' in raw.columns:
        raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        raw['row_id'] = raw.index
        
    df = splits.merge(raw[['row_id', 'text']], on='row_id', how='left')
    df['text'] = df['text'].fillna('')
    
    # We already have flags and length_tokens in splits_8class.csv
    # But we need num sentences.
    print("Extracting extra features...")
    df['num_sentences'] = df['text'].apply(lambda x: len([s for s in re.split(r'[.!?]+', str(x)) if s.strip()]) if x else 0)
    df['char_length'] = df['text'].apply(len)
    df['tokens_exceed_128'] = (df['length_tokens'] > 128).astype(int)
    df['first_3_words'] = df['text'].apply(lambda x: " ".join(str(x).split()[:3]))
    
    # Encode labels
    # 8 classes
    unique_labels = df['multi_label'].unique()
    label_map = {lbl: i for i, lbl in enumerate(sorted(unique_labels))}
    df['target'] = df['multi_label'].map(label_map)
    
    report = ["# Shortcut Baselines Report\n"]
    report.append("Macro-F1 (chance ≈ 0.125) on val and test sets using simple shortcut features.\n")
    report.append("| Cluster | Model | Val F1 | Test F1 | Test F1 (Clean Subset) |")
    report.append("|---|---|---|---|---|")
    
    for cluster in df['cluster'].unique():
        if cluster == 'unknown': continue
        c_df = df[df['cluster'] == cluster]
        
        train = c_df[c_df['split'] == 'train']
        val = c_df[c_df['split'] == 'val']
        test = c_df[c_df['split'] == 'test']
        
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            continue
            
        print(f"Training for cluster {cluster}...")
        
        # Define features
        def get_len_feats(data):
            return data[['char_length', 'length_tokens', 'tokens_exceed_128', 'num_sentences']].values
            
        def get_flag_feats(data):
            return data[['flag_junk', 'flag_script_mismatch', 'flag_prompt_echo']].astype(int).values
            
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf.fit(train['first_3_words'])
        
        def get_tfidf_feats(data):
            return tfidf.transform(data['first_3_words'])
            
        # Target
        y_train = train['target'].values
        y_val = val['target'].values
        y_test = test['target'].values
        
        # Clean subset mask for test
        test_clean_mask = (test['flag_junk'] == False) & (test['flag_script_mismatch'] == False) & (test['flag_prompt_echo'] == False)
        
        def evaluate(X_tr, X_va, X_te, name):
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_tr, y_train)
            pred_va = clf.predict(X_va)
            pred_te = clf.predict(X_te)
            
            f1_va = f1_score(y_val, pred_va, average='macro')
            f1_te = f1_score(y_test, pred_te, average='macro')
            
            if test_clean_mask.sum() > 0:
                f1_te_clean = f1_score(y_test[test_clean_mask], pred_te[test_clean_mask], average='macro')
                clean_str = f"{f1_te_clean:.3f}"
            else:
                clean_str = "N/A"
                
            report.append(f"| {cluster} | {name} | {f1_va:.3f} | {f1_te:.3f} | {clean_str} |")

        # a) length features
        evaluate(get_len_feats(train), get_len_feats(val), get_len_feats(test), "(a) Length Features")
        
        # b) TF-IDF first 3 words
        evaluate(get_tfidf_feats(train), get_tfidf_feats(val), get_tfidf_feats(test), "(b) First 3 Words TF-IDF")
        
        # c) flags only
        evaluate(get_flag_feats(train), get_flag_feats(val), get_flag_feats(test), "(c) Flags Only")
        
        # d) all combined
        X_tr_all = hstack([csr_matrix(get_len_feats(train)), get_tfidf_feats(train), csr_matrix(get_flag_feats(train))])
        X_va_all = hstack([csr_matrix(get_len_feats(val)), get_tfidf_feats(val), csr_matrix(get_flag_feats(val))])
        X_te_all = hstack([csr_matrix(get_len_feats(test)), get_tfidf_feats(test), csr_matrix(get_flag_feats(test))])
        
        evaluate(X_tr_all, X_va_all, X_te_all, "(d) All Combined")
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Done. Saved to {out_path}")

if __name__ == '__main__':
    shortcut_baselines()
