import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)
import pandas as pd
import os
import sys
import numpy as np
from transformers import AutoTokenizer
import regex
from collections import Counter
import hashlib
import string
import re

from clusters import LANG_TO_CLUSTER, SCRIPT_CLUSTERS

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

def detect_dominant_script(text):
    # Regex to detect scripts
    script_counts = {
        'latin': len(regex.findall(r'\p{IsLatin}', text)),
        'cyrillic': len(regex.findall(r'\p{IsCyrillic}', text)),
        'greek': len(regex.findall(r'\p{IsGreek}', text)),
        'arabic': len(regex.findall(r'\p{IsArabic}', text)),
        'hanzi': len(regex.findall(r'\p{IsHan}', text))
    }
    total = sum(script_counts.values())
    if total == 0:
        return 'unknown'
    return max(script_counts, key=script_counts.get)

def audit_shortcuts():
    raw_path = 'data/raw/multitude_v3_clean.csv'
    out_path = 'results/shortcut_audit.md'
    
    if not os.path.exists(raw_path):
        print("Raw data not found")
        sys.exit(1)
        
    print("Loading data...")
    df = pd.read_csv(raw_path)
    
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        df['row_id'] = df.index
        
    df['cluster'] = df['language'].apply(lambda x: LANG_TO_CLUSTER.get(x, 'unknown'))
    
    # Filter to base paper 18 to align with our scope
    from clusters import BASE_PAPER_18_LANGUAGES
    df = df[df['language'].isin(BASE_PAPER_18_LANGUAGES)].copy()
    
    print("Tokenizing...")
    # Load tokenizer without weights
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    
    # Only take a subset for speed if necessary, but it asks for "Per cluster per generator", meaning all
    df['text'] = df['text'].fillna('')
    df['char_length'] = df['text'].apply(len)
    
    # Apply tokenization
    df['tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True)))
    
    # Dominant script
    df['dominant_script'] = df['text'].apply(detect_dominant_script)
    df['script_mismatch'] = df['dominant_script'] != df['cluster']
    
    # Opening 3 words
    df['first_3_words'] = df['text'].apply(lambda x: " ".join(normalize_text(x).split()[:3]))
    
    # Boilerplate patterns
    boilerplate = re.compile(r'(as an ai|i\'m sorry|i am sorry|sure|here is|here are|certainly|unfortunately)', re.IGNORECASE)
    df['has_boilerplate'] = df['text'].apply(lambda x: 1 if boilerplate.search(str(x)[:200]) else 0)
    
    report = ["# Shortcut Audit Report\n"]
    
    for cluster in df['cluster'].unique():
        if cluster == 'unknown': continue
        report.append(f"\n## Cluster: {cluster}")
        c_df = df[df['cluster'] == cluster]
        
        for gen in c_df['multi_label'].unique():
            g_df = c_df[c_df['multi_label'] == gen]
            if len(g_df) == 0: continue
            
            chars = g_df['char_length']
            toks = g_df['tokens']
            
            mean_char = chars.mean()
            p10_char = chars.quantile(0.10)
            p90_char = chars.quantile(0.90)
            
            mean_tok = toks.mean()
            p10_tok = toks.quantile(0.10)
            p90_tok = toks.quantile(0.90)
            
            over_128 = (toks > 128).mean() * 100
            mismatch_rate = g_df['script_mismatch'].mean() * 100
            boiler_rate = g_df['has_boilerplate'].mean() * 100
            
            top_3_words = Counter(g_df['first_3_words']).most_common(10)
            top_3_str = ", ".join([f"'{k}' ({v})" for k,v in top_3_words])
            
            report.append(f"### Generator: {gen}")
            report.append(f"- **Chars**: mean {mean_char:.1f}, p10 {p10_char:.1f}, p90 {p90_char:.1f}")
            report.append(f"- **Tokens**: mean {mean_tok:.1f}, p10 {p10_tok:.1f}, p90 {p90_tok:.1f} ({over_128:.1f}% > 128)")
            report.append(f"- **Script Mismatch Rate**: {mismatch_rate:.2f}%")
            report.append(f"- **Boilerplate Rate**: {boiler_rate:.2f}%")
            report.append(f"- **Top Openings**: {top_3_str}")
            
    # Find 17 overlapping hashes
    report.append("\n## Overlapping Train-Test Rows (Leakage)")
    df['norm_hash'] = df['text'].apply(lambda x: hash_text(normalize_text(x)))
    test_hashes = set(df[df['split'] == 'test']['norm_hash'])
    train_overlap = df[(df['split'] == 'train') & (df['norm_hash'].isin(test_hashes))]
    
    report.append(f"Found {len(train_overlap)} train rows overlapping with test.")
    for idx, row in train_overlap.iterrows():
        short_text = str(row['text']).replace('\n', ' ')[:100]
        report.append(f"- Row ID {row['row_id']} ({row['language']}, {row['multi_label']}): {short_text}...")
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Done. Saved to {out_path}")

if __name__ == '__main__':
    audit_shortcuts()
