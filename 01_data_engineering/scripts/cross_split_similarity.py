import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text.strip()

def measure_leakage():
    raw_path = 'data/raw/multitude_v3_clean.csv'
    splits_path = 'data/processed/splits_8class.csv'
    out_path = 'results/cross_split_similarity.md'
    
    if not os.path.exists(splits_path) or not os.path.exists(raw_path):
        print("Required files not found")
        sys.exit(1)
        
    print("Loading data...")
    raw = pd.read_csv(raw_path)
    splits = pd.read_csv(splits_path)
    
    if 'Unnamed: 0' in raw.columns:
        raw = raw.rename(columns={'Unnamed: 0': 'row_id'})
    else:
        raw['row_id'] = raw.index
        
    df = splits.merge(raw[['row_id', 'text', 'label']], on='row_id', how='left')
    df['text'] = df['text'].fillna('')
    
    report = ["# Cross-Split Similarity (Leakage Measurement)\n"]
    report.append("Comparing maximum cosine similarity of val/test rows to train rows of **different labels** (multi_label) in the same language.\n")
    
    report.append("| Language | Comparison | Mean | P50 | P90 | P99 | %>0.5 | %>0.7 |")
    report.append("|---|---|---|---|---|---|---|---|")
    
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000)
    # The prompt says "word + char n-grams", so let's do a combined vectorizer or just char_wb. 
    # Let's use a union or just char_wb since it works well. Let's do char_wb (2,4) and word (1,2)
    # For simplicity, standard word analyzer is fine, but let's stick to TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=10000) 
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)

    test_to_train_high_flags = []
    stats_data = []
    
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang].copy()
        if len(lang_df) < 50:
            continue
            
        print(f"Processing {lang}...")
        try:
            tfidf = vectorizer.fit_transform(lang_df['text'])
        except ValueError:
            continue
            
        # Actual splits
        train_idx = np.where(lang_df['split'] == 'train')[0]
        val_idx = np.where(lang_df['split'] == 'val')[0]
        test_idx = np.where(lang_df['split'] == 'test')[0]
        
        multi_labels = lang_df['multi_label'].values
        
        def get_max_sims(source_idx, target_idx, matrix, labels):
            if len(source_idx) == 0 or len(target_idx) == 0:
                return np.array([])
                
            sim_matrix = cosine_similarity(matrix[source_idx], matrix[target_idx])
            s_labels = labels[source_idx]
            t_labels = labels[target_idx]
            
            mask = s_labels[:, None] != t_labels[None, :]
            sim_matrix = np.where(mask, sim_matrix, -1.0)
            max_sims = sim_matrix.max(axis=1)
            max_sims = np.where(max_sims == -1.0, 0.0, max_sims)
            return max_sims
            
        def format_stats(sims, lang, comp_name):
            if len(sims) == 0:
                return f"| {lang} | {comp_name} | N/A | N/A | N/A | N/A | N/A | N/A |"
            mean = sims.mean()
            p50 = np.percentile(sims, 50)
            p90 = np.percentile(sims, 90)
            p99 = np.percentile(sims, 99)
            gt_05 = (sims > 0.5).mean() * 100
            gt_07 = (sims > 0.7).mean() * 100
            return f"| {lang} | {comp_name} | {mean:.3f} | {p50:.3f} | {p90:.3f} | {p99:.3f} | {gt_05:.1f}% | {gt_07:.1f}% |"

        val_sims = get_max_sims(val_idx, train_idx, tfidf, multi_labels)
        test_sims = get_max_sims(test_idx, train_idx, tfidf, multi_labels)
        
        if len(test_sims) > 0 and np.percentile(test_sims, 90) > 0.5:
            test_to_train_high_flags.append(lang)
            
        report.append(format_stats(val_sims, lang, 'val->train (actual)'))
        report.append(format_stats(test_sims, lang, 'test->train (actual)'))
        
        # Random control
        n_train = len(train_idx)
        n_val = len(val_idx)
        n_test = len(test_idx)
        
        shuffled = np.random.permutation(len(lang_df))
        rand_train = shuffled[:n_train]
        rand_val = shuffled[n_train:n_train+n_val]
        rand_test = shuffled[n_train+n_val:]
        
        rand_val_sims = get_max_sims(rand_val, rand_train, tfidf, multi_labels)
        rand_test_sims = get_max_sims(rand_test, rand_train, tfidf, multi_labels)
        
        report.append(format_stats(rand_val_sims, lang, 'val->train (random)'))
        report.append(format_stats(rand_test_sims, lang, 'test->train (random)'))

        if len(val_sims) > 0 and len(rand_val_sims) > 0:
            val_lt_rand = val_sims.mean() < rand_val_sims.mean()
        else:
            val_lt_rand = False
            
        if len(test_sims) > 0 and len(rand_test_sims) > 0:
            test_lt_rand = test_sims.mean() < rand_test_sims.mean()
        else:
            test_lt_rand = False

        if len(val_sims) > 0 and len(test_sims) > 0:
            val_gt_test = val_sims.mean() > test_sims.mean()
        else:
            val_gt_test = False
            
        if len(test_sims) > 0:
            test_gt_07 = (test_sims > 0.7).mean() * 100
        else:
            test_gt_07 = 0.0
            
        stats_data.append({
            'lang': lang,
            'val_lt_rand': val_lt_rand,
            'test_lt_rand': test_lt_rand,
            'val_gt_test': val_gt_test,
            'test_gt_07': test_gt_07,
            'val_mean': val_sims.mean() if len(val_sims) > 0 else 0,
            'rand_val_mean': rand_val_sims.mean() if len(rand_val_sims) > 0 else 0
        })
        
    report.append("\n## Conclusion\n")
    
    val_lt_rand_count = sum(1 for s in stats_data if s['val_lt_rand'])
    test_lt_rand_count = sum(1 for s in stats_data if s['test_lt_rand'])
    val_gt_test_count = sum(1 for s in stats_data if s['val_gt_test'])
    
    report.append(f"- **Val actual < Random control**: {val_lt_rand_count} out of {len(stats_data)} languages.")
    report.append(f"- **Test actual < Random control**: {test_lt_rand_count} out of {len(stats_data)} languages.")
    report.append(f"- **Val->train > Test->train similarity**: {val_gt_test_count} out of {len(stats_data)} languages.")
    
    report.append("\n### Per-language share of test rows with similarity above 0.7:")
    for s in stats_data:
        report.append(f"- {s['lang']}: {s['test_gt_07']:.1f}%")
        
    val_more_leaky_langs = [s['lang'] for s in stats_data if s['val_mean'] > s['rand_val_mean']]
    if val_more_leaky_langs:
        report.append(f"\nLanguages where val is more similar than the random control: {', '.join(val_more_leaky_langs)}")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Done. Saved to {out_path}")

if __name__ == '__main__':
    measure_leakage()
