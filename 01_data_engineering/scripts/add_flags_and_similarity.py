import argparse
import pandas as pd
import numpy as np
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os

def is_hanzi_junk(text):
    cjk_kana = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309F\u30A0-\u30FF]', str(text)))
    return cjk_kana < 5

def is_other_junk(text):
    letters = sum(1 for c in str(text) if c.isalpha())
    return letters < 20

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='data/raw/multitude_v3_clean.csv')
    parser.add_argument('--splits_path', default='data/processed/splits_8class.csv')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    splits = pd.read_csv(args.splits_path)
    raw = pd.read_csv(args.raw_path)

    if 'Unnamed: 0' in raw.columns:
        raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        raw['row_id'] = raw.index

    df = splits.merge(raw[['row_id', 'text', 'label']], on='row_id', how='left')
    df['text'] = df['text'].fillna('')

    # 1. Update flag_junk
    hanzi_mask = df['cluster'] == 'hanzi'
    other_mask = ~hanzi_mask

    new_junk = np.zeros(len(df), dtype=bool)
    new_junk[hanzi_mask] = df[hanzi_mask]['text'].apply(is_hanzi_junk).values
    new_junk[other_mask] = df[other_mask]['text'].apply(is_other_junk).values

    df['flag_junk'] = new_junk
    splits['flag_junk'] = new_junk

    # 2. Add max_sim_train_other and story_disjoint
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)

    max_sim = np.full(len(df), np.nan)
    story_disjoint = np.zeros(len(df), dtype=bool)

    for lang in df['language'].unique():
        idx = df[df['language'] == lang].index
        if len(idx) == 0:
            continue
        
        texts = df.loc[idx, 'text'].values
        splits_l = df.loc[idx, 'split'].values
        labels_l = df.loc[idx, 'multi_label'].values
        
        try:
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:
            continue
            
        train_local_idx = np.where(splits_l == 'train')[0]
        val_test_local_idx = np.where(np.isin(splits_l, ['val', 'test']))[0]
        
        if len(train_local_idx) == 0 or len(val_test_local_idx) == 0:
            continue
            
        sim_matrix = cosine_similarity(tfidf[val_test_local_idx], tfidf[train_local_idx])
        
        vt_labels = labels_l[val_test_local_idx]
        tr_labels = labels_l[train_local_idx]
        
        mask = vt_labels[:, None] != tr_labels[None, :]
        sim_matrix = np.where(mask, sim_matrix, -1.0)
        
        m_sims = sim_matrix.max(axis=1)
        m_sims = np.where(m_sims == -1.0, 0.0, m_sims)
        
        global_idx = idx[val_test_local_idx]
        max_sim[global_idx] = m_sims
        story_disjoint[global_idx] = m_sims < 0.7

    splits['max_sim_train_other'] = max_sim
    splits['story_disjoint'] = np.where(splits['split'] == 'train', pd.NA, story_disjoint)

    splits.to_csv(args.splits_path, index=False)
    print(f"Updated {args.splits_path} with new flag_junk, max_sim_train_other, story_disjoint.")

if __name__ == '__main__':
    main()
