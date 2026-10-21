"""
evaluate_cross_cluster.py
-------------------------
Evaluates a trained cluster's adapter and classifier on any test cluster's data
(supports in-cluster and cross-cluster/zero-shot transfer evaluation).

Usage:
    python 03_inference_evaluation/src/evaluate_cross_cluster.py \
        --data data/raw/multitude_v3.csv \
        --train_cluster latin \
        --test_cluster cyrillic \
        --eval_adapter latin
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

from model import SFACAModel, build_tokenizer
from data_loader import load_multitude_csv, filter_by_clusters, filter_by_split, AttributionDataset
from clusters import all_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the MULTITuDE CSV file.")
    parser.add_argument("--train_cluster", type=str, required=True, choices=all_clusters(),
                        help="The cluster whose adapter/classifier we want to load.")
    parser.add_argument("--test_cluster", type=str, required=True, choices=all_clusters(),
                        help="The cluster we want to evaluate on.")
    parser.add_argument("--eval_adapter", type=str, default=None,
                        help="Which adapter to activate during evaluation. Defaults to train_cluster.")
    parser.add_argument("--adapter_dir", type=str, default="results/adapters",
                        help="Directory where adapters are saved.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_adapter = args.eval_adapter or args.train_cluster
    print(f"Loading model trained on '{args.train_cluster}'...")
    print(f"Evaluating on test split of '{args.test_cluster}' using adapter '{eval_adapter}'...")

    # 1. Load Data
    df = load_multitude_csv(args.data)
    test_df = filter_by_clusters(df, [args.test_cluster])
    test_df = filter_by_split(test_df, "test")
    print(f"Loaded {len(test_df)} test samples for cluster '{args.test_cluster}'.")

    if len(test_df) == 0:
        print(f"Warning: No test samples found for cluster '{args.test_cluster}'!")
        return

    tokenizer = build_tokenizer()
    test_ds = AttributionDataset(test_df, tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Build Model
    model = SFACAModel(clusters=all_clusters()).to(device)

    # 3. Load trained adapter weights
    adapter_path = os.path.abspath(os.path.join(args.adapter_dir, args.train_cluster))
    adapter_file = os.path.join(adapter_path, "adapter_model.bin")
    
    if os.path.exists(adapter_file):
        print(f"Loading PEFT adapter weights from {adapter_file}...")
        from peft import set_peft_model_state_dict
        adapter_weights = torch.load(adapter_file, map_location=device)
        set_peft_model_state_dict(model.backbone, adapter_weights, adapter_name=args.train_cluster)
    else:
        print(f"Notice: PEFT adapter file not found in {adapter_path}.")

    # 4. Load trained classifier state dict
    classifier_path = os.path.join(adapter_path, "classifier.pt")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier weights not found at {classifier_path}. Make sure train.py has saved them.")
    
    print(f"Loading classifier weights from {classifier_path}...")
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    # 5. Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask, eval_adapter)
            preds = logits.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # 6. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n================ RESULTS FOR {args.test_cluster} (using {eval_adapter} adapter) ================")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print("========================================================================\n")

    print("Classification Report:")
    from data_loader import ID2LABEL
    unique_labels = sorted(list(set(all_labels) | set(all_preds)))
    target_names = [ID2LABEL[i] for i in unique_labels]
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # Save to results csv
    results_dir = "results/eval"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"eval_{args.train_cluster}_on_{args.test_cluster}_via_{eval_adapter}.csv")
    
    res_df = pd.DataFrame([{
        "train_cluster": args.train_cluster,
        "test_cluster": args.test_cluster,
        "eval_adapter": eval_adapter,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }])
    res_df.to_csv(results_file, index=False)
    print(f"Saved evaluation results to {results_file}")


if __name__ == "__main__":
    main()
