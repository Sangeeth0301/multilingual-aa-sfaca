"""
low_resource_eval.py
--------------------
Zero-Shot Evaluation module for unseen Low-Resource Languages (Phase 4 / Exp 3).

Evaluates how SFA-CA script-family adapters perform on low-resource languages
(e.g., Tamil 'ta', Swahili 'sw', Amharic 'am', Hindi 'hi') that were NOT present
in the base training set, validating zero-shot cross-script generalization.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import argparse
import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer

from clusters import nearest_cluster_for_unseen, LOW_RESOURCE_LANGUAGES
from data_loader import load_multitude_csv, filter_by_split, AttributionDataset, ID2LABEL
from model import SFACAModel


def evaluate_low_resource(
    data_path: str,
    target_lang: str,
    script_hint: str,
    adapters_dir: str = "results/adapters",
    batch_size: int = 16,
    max_length: int = 256
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine nearest script cluster for zero-shot transfer
    active_cluster = nearest_cluster_for_unseen(target_lang, script_hint)
    print(f"\n=======================================================")
    print(f"Zero-Shot Evaluation for Low-Resource Language: '{target_lang}'")
    print(f"Mapped Script Hint: '{script_hint}' -> Active Adapter: '{active_cluster}'")
    print(f"=======================================================\n")

    # Load dataset with restrict_to_base_paper_18=False
    df = load_multitude_csv(data_path, restrict_to_base_paper_18=False)
    available_langs = df["language"].unique().tolist()
    
    # If requested lang not found or "auto", evaluate all unseen languages in dataset
    if target_lang not in available_langs or target_lang == "auto":
        unseen_langs = [l for l in available_langs if l not in ["nl", "en", "de", "el", "ar", "zh", "bg", "uk", "ru", "hr", "cs", "pl", "sk", "sl", "pt", "ro", "es", "hu"]]
        if not unseen_langs:
            unseen_langs = [available_langs[0]]
        print(f"Notice: '{target_lang}' not in dataset. Evaluating available unseen languages: {unseen_langs}")
        
        all_results = []
        for ul in unseen_langs:
            res = evaluate_low_resource(data_path, ul, script_hint, adapters_dir, batch_size, max_length)
            if res is not None:
                all_results.append(res)
        return pd.concat(all_results, ignore_index=True) if all_results else None

    lang_df = df[df["language"] == target_lang].reset_index(drop=True)
    if len(lang_df) == 0:
        print(f"No samples found in dataset for language '{target_lang}'.")
        return None

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    dataset = AttributionDataset(lang_df, tokenizer, max_length=max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SFACAModel().to(device)

    # Load adapter
    adapter_path = os.path.join(adapters_dir, active_cluster)
    adapter_file = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.exists(adapter_file):
        from peft import set_peft_model_state_dict
        adapter_weights = torch.load(adapter_file, map_location=device)
        set_peft_model_state_dict(model.backbone, adapter_weights, adapter_name=active_cluster)
        print(f"Loaded PEFT adapter weights for '{active_cluster}'")
        
    classifier_path = os.path.join(adapter_path, "classifier.pt")
    if os.path.exists(classifier_path):
        model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print(f"Loaded classifier weights for '{active_cluster}'")

    model.eval()
    all_preds, all_labels = [], []

    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask, active_cluster)
            preds = logits.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    results_dir = "results/low_resource"
    os.makedirs(results_dir, exist_ok=True)
    res_path = os.path.join(results_dir, f"zero_shot_{target_lang}_via_{active_cluster}.csv")
    
    res_df = pd.DataFrame([{
        "language": target_lang,
        "script_hint": script_hint,
        "active_adapter": active_cluster,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }])
    res_df.to_csv(res_path, index=False)
    print(f"Saved zero-shot results to {res_path}")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot low-resource evaluation")
    parser.add_argument("--data", type=str, required=True, help="Path to MULTITuDE CSV file")
    parser.add_argument("--lang", "--target_lang", type=str, default="ta", help="Target low-resource ISO language code")
    parser.add_argument("--script_hint", type=str, default="latin", help="Script family hint")
    parser.add_argument("--adapter_dir", "--adapters_dir", type=str, default="results/adapters", help="Directory containing trained adapters")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    args = parser.parse_args()

    evaluate_low_resource(
        data_path=args.data,
        target_lang=args.lang,
        script_hint=args.script_hint,
        adapters_dir=args.adapter_dir,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
