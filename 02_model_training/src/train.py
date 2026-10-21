"""
train.py
--------
Training pipeline for SFA-CA (Script-Family Adapter with Contrastive Attribution).
Supports:
- Per-cluster LoRA training ("latin", "cyrillic", "greek", "arabic", "hanzi")
- Monolithic multi-cluster baseline ("all")
- Seeded, class-stratified subsampling (--max_train_samples)
- Smoke testing (--smoke)
- Shuffled-label sanity testing (--shuffle_labels)
- Mid-run resume from last completed epoch with exact parity
- Val-only model selection with single final test evaluation
- Metric logging to metrics.json and completion marking with DONE.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from clusters import all_clusters
from contrastive_loss import combined_loss
from data_loader import (
    FROZEN_SPLITS_SHA256,
    LABEL_LIST_8CLASS,
    AttributionDataset,
    compute_file_sha256,
    filter_by_clusters,
    filter_by_split,
    load_training_table,
    stratified_subsample,
)
from model import LORA_TARGET_MODULES, SFACAModel, build_tokenizer


def set_seeds(seed: int):
    """Seed all pseudo-random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def get_package_versions() -> dict:
    import peft
    import sklearn
    import transformers

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    df_meta: pd.DataFrame = None,
) -> dict:
    """Computes comprehensive macro-F1, accuracy, per-class F1, confusion matrix, subset F1, and AUROC."""
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    per_class_f1 = [float(x) for x in f1_score(y_true, preds, average=None)]
    cm = confusion_matrix(y_true, preds).tolist()

    # Binary human-vs-AI: class 0 is human, classes 1..7 are AI
    binary_labels = (y_true > 0).astype(int)
    p_ai = 1.0 - probs[:, 0]
    binary_preds = (p_ai >= 0.5).astype(int)
    binary_f1 = float(f1_score(binary_labels, binary_preds))

    if len(np.unique(binary_labels)) > 1:
        binary_auroc = float(roc_auc_score(binary_labels, p_ai))
    else:
        binary_auroc = None

    metrics = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "binary_human_vs_ai": {
            "f1": binary_f1,
            "auroc": binary_auroc,
        },
    }

    if df_meta is not None:
        # 1. story_disjoint == True
        if "story_disjoint" in df_meta.columns:
            mask_disjoint = df_meta["story_disjoint"].values == True
            if mask_disjoint.sum() > 0:
                metrics["macro_f1_story_disjoint"] = float(
                    f1_score(y_true[mask_disjoint], preds[mask_disjoint], average="macro")
                )

        # 2. flags_clean: flag_junk == False & flag_script_mismatch == False & flag_prompt_echo == False
        flags = ["flag_junk", "flag_script_mismatch", "flag_prompt_echo"]
        if all(f in df_meta.columns for f in flags):
            mask_clean = (
                (df_meta["flag_junk"].values == False)
                & (df_meta["flag_script_mismatch"].values == False)
                & (df_meta["flag_prompt_echo"].values == False)
            )
            if mask_clean.sum() > 0:
                metrics["macro_f1_flags_clean"] = float(
                    f1_score(y_true[mask_clean], preds[mask_clean], average="macro")
                )

            # 3. disjoint and flags clean
            if "story_disjoint" in df_meta.columns:
                mask_both = mask_disjoint & mask_clean
                if mask_both.sum() > 0:
                    metrics["macro_f1_disjoint_and_flags_clean"] = float(
                        f1_score(y_true[mask_both], preds[mask_both], average="macro")
                    )

    return metrics


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    adapter_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Evaluates model over a DataLoader and returns (probs, labels, row_ids)."""
    model.eval()
    all_probs, all_labels, all_row_ids = [], [], []
    use_amp = (device.type == "cuda")

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        row_ids = batch["row_id"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids, attention_mask, cluster_name=adapter_name)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_labels.extend(labels.tolist())
        all_row_ids.extend(row_ids if isinstance(row_ids, list) else row_ids.tolist())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.array(all_labels)
    return probs_arr, labels_arr, all_row_ids


def train_run(args):
    """Main training execution function."""
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_dir = os.path.join(args.out_dir, f"{args.cluster}_seed{args.seed}{tag_suffix}")
    done_path = os.path.join(run_dir, "DONE.json")

    # 1. Skip if already finished
    if os.path.exists(done_path):
        print(f"[SKIP] Run in {run_dir} is already completed (found DONE.json).")
        with open(done_path, "r") as f:
            return json.load(f)

    os.makedirs(run_dir, exist_ok=True)
    set_seeds(args.seed)

    # Device selection
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_name = torch.cuda.get_device_name(0) if (device.type == "cuda" and torch.cuda.is_available()) else "CPU"
    print(f"Device: {device} | GPU: {gpu_name}")

    # 2. Load data and verify hash
    merged_df, splits_sha256 = load_training_table(
        raw_path=args.raw_data,
        splits_path=args.splits_data,
        allow_new_splits=args.allow_new_splits,
    )
    raw_data_sha256 = compute_file_sha256(args.raw_data)

    # Filter cluster
    adapter_name = args.cluster
    if args.cluster == "all":
        # Monolithic baseline across all 5 clusters
        active_clusters = ["latin", "cyrillic", "greek", "arabic", "hanzi"]
        cluster_df = filter_by_clusters(merged_df, active_clusters)
    else:
        cluster_df = filter_by_clusters(merged_df, [args.cluster])

    train_df = filter_by_split(cluster_df, "train").copy()
    val_df = filter_by_split(cluster_df, "val").copy()
    test_df = filter_by_split(cluster_df, "test").copy()

    # Smoke mode override
    if args.smoke:
        args.epochs = 1
        args.max_train_samples = 500
        print("[SMOKE MODE] Running 1 epoch with 500 class-stratified samples.")

    # Stratified subsampling
    if args.max_train_samples is not None and len(train_df) > args.max_train_samples:
        train_df = stratified_subsample(train_df, max_samples=args.max_train_samples, seed=args.seed)
        print(f"Subsampled train set to {len(train_df)} samples (stratified, seed={args.seed})")

    # Shuffled labels sanity check (train labels only!)
    if args.shuffle_labels:
        print("[SANITY] Shuffling TRAIN labels only (val and test remain unchanged).")
        rng = np.random.RandomState(args.seed)
        train_df["label_id"] = rng.permutation(train_df["label_id"].values)

    print(f"Dataset sizes: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # 3. Model & Tokenizer
    tokenizer = build_tokenizer(args.backbone)

    train_ds = AttributionDataset(train_df, tokenizer, max_length=args.max_length)
    val_ds = AttributionDataset(val_df, tokenizer, max_length=args.max_length)
    test_ds = AttributionDataset(test_df, tokenizer, max_length=args.max_length)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SFACAModel(
        backbone_name=args.backbone,
        num_classes=len(LABEL_LIST_8CLASS),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        clusters=[adapter_name],
    ).to(device)
    model.set_active_cluster(adapter_name)

    # Trainable params: adapter + classifier
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_count:,} (LoRA targets: {LORA_TARGET_MODULES})")
    assert trainable_count > 0, "No trainable parameters found!"

    # Optimizer, Scheduler, Scaler
    optimizer = AdamW(trainable_params, lr=args.lr)

    total_training_steps = len(train_loader) * args.epochs
    if args.lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        scheduler = None

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # WandB
    use_wandb = args.wandb and ("WANDB_API_KEY" in os.environ)
    if use_wandb:
        import wandb

        wandb.init(
            project="sfa-ca-multilingual-attribution",
            name=f"{args.cluster}-seed{args.seed}",
            config=vars(args),
        )

    # Check for mid-run resume
    resume_path = os.path.join(run_dir, "resume_state.json")
    start_epoch = 0
    best_val_f1 = -1.0
    best_epoch = -1
    best_val_acc = 0.0
    epoch_metrics = []

    if os.path.exists(resume_path) and os.path.exists(os.path.join(run_dir, "optimizer.pt")):
        print(f"[RESUME] Found existing checkpoint in {run_dir}. Resuming...")
        with open(resume_path, "r") as f:
            resume_data = json.load(f)

        start_epoch = resume_data["completed_epoch"]
        best_val_f1 = resume_data["best_val_f1"]
        best_epoch = resume_data["best_epoch"]
        best_val_acc = resume_data.get("best_val_acc", 0.0)
        epoch_metrics = resume_data.get("epoch_metrics", [])

        # Load weights
        adapter_state = torch.load(
            os.path.join(run_dir, "adapter_model_last.pt"),
            map_location=device,
            weights_only=True,
        )
        set_peft_model_state_dict(model.backbone, adapter_state, adapter_name=adapter_name)

        classifier_state = torch.load(
            os.path.join(run_dir, "classifier_last.pt"),
            map_location=device,
            weights_only=True,
        )
        model.classifier.load_state_dict(classifier_state)

        optimizer.load_state_dict(
            torch.load(os.path.join(run_dir, "optimizer.pt"), map_location=device, weights_only=True)
        )

        if scheduler is not None and os.path.exists(os.path.join(run_dir, "scheduler.pt")):
            scheduler.load_state_dict(
                torch.load(os.path.join(run_dir, "scheduler.pt"), map_location=device, weights_only=True)
            )

        if os.path.exists(os.path.join(run_dir, "scaler.pt")):
            scaler.load_state_dict(
                torch.load(os.path.join(run_dir, "scaler.pt"), map_location=device, weights_only=True)
            )

        if os.path.exists(os.path.join(run_dir, "rng_state.pt")):
            rng = torch.load(os.path.join(run_dir, "rng_state.pt"), weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

        print(f"[RESUME] Resuming from epoch {start_epoch + 1}/{args.epochs} (best_val_f1={best_val_f1:.4f})")

    total_train_start = time.time()

    # 4. Training loop (epoch-by-epoch)
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = model.get_embedding(input_ids, attention_mask, adapter_name)
                logits = model.classifier(embeddings)
                loss = combined_loss(
                    logits,
                    embeddings,
                    labels,
                    contrastive_weight=args.contrastive_weight,
                    temperature=args.temperature,
                )

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        epoch_seconds = time.time() - epoch_start_time
        samples_per_sec = len(train_df) / epoch_seconds if epoch_seconds > 0 else 0.0
        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on VAL ONLY (never touch test split here!)
        val_probs, val_labels, val_row_ids = evaluate_dataset(model, val_loader, adapter_name, device)
        val_metrics = compute_metrics(val_labels, val_probs, val_df)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "seconds": round(epoch_seconds, 2),
            "samples_per_sec": round(samples_per_sec, 2),
        }
        epoch_metrics.append(epoch_record)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_train_loss:.4f} | "
            f"Val Macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
            f"{samples_per_sec:.1f} samples/s ({epoch_seconds:.1f}s)"
        )

        if use_wandb:
            wandb.log(epoch_record)

        # Check if validation improved
        adapter_state = get_peft_model_state_dict(model.backbone, adapter_name=adapter_name)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"  --> Saved new best checkpoint (Val Macro-F1: {best_val_f1:.4f})")
            torch.save(adapter_state, os.path.join(run_dir, "adapter_model.bin"))
            torch.save(model.classifier.state_dict(), os.path.join(run_dir, "classifier.pt"))
            torch.save(model.classifier.state_dict(), os.path.join(run_dir, "classifier_head.pt"))
            np.save(os.path.join(run_dir, "val_probs.npy"), val_probs)
            np.save(os.path.join(run_dir, "val_row_ids.npy"), np.array(val_row_ids))

        # Save last checkpoint for mid-run resume
        torch.save(adapter_state, os.path.join(run_dir, "adapter_model_last.pt"))
        torch.save(model.classifier.state_dict(), os.path.join(run_dir, "classifier_last.pt"))
        torch.save(optimizer.state_dict(), os.path.join(run_dir, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(run_dir, "scheduler.pt"))
        torch.save(scaler.state_dict(), os.path.join(run_dir, "scaler.pt"))

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, os.path.join(run_dir, "rng_state.pt"))

        resume_metadata = {
            "completed_epoch": epoch + 1,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "epoch_metrics": epoch_metrics,
        }
        with open(resume_path, "w") as f:
            json.dump(resume_metadata, f, indent=2)

    total_train_seconds = time.time() - total_train_start
    overall_samples_per_sec = (len(train_df) * (args.epochs - start_epoch)) / max(total_train_seconds, 1e-5)

    # 5. Load BEST model checkpoint for FINAL TEST EVALUATION (performed ONCE)
    print(f"\nTraining complete. Loading best checkpoint from Epoch {best_epoch} for test evaluation...")
    best_adapter_path = os.path.join(run_dir, "adapter_model.bin")
    best_classifier_path = os.path.join(run_dir, "classifier.pt")
    if os.path.exists(best_adapter_path) and os.path.exists(best_classifier_path):
        best_adapter = torch.load(best_adapter_path, map_location=device, weights_only=True)
        set_peft_model_state_dict(model.backbone, best_adapter, adapter_name=adapter_name)
        best_classifier = torch.load(best_classifier_path, map_location=device, weights_only=True)
        model.classifier.load_state_dict(best_classifier)

    # Final Val Evaluation with best checkpoint
    final_val_probs, final_val_labels, final_val_row_ids = evaluate_dataset(model, val_loader, adapter_name, device)
    final_val_metrics = compute_metrics(final_val_labels, final_val_probs, val_df)

    # Final Test Evaluation (EXACTLY ONCE)
    print("Evaluating on test split...")
    test_probs, test_labels, test_row_ids = evaluate_dataset(model, test_loader, adapter_name, device)
    final_test_metrics = compute_metrics(test_labels, test_probs, test_df)

    np.save(os.path.join(run_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(run_dir, "test_row_ids.npy"), np.array(test_row_ids))

    # Per-cluster breakdown (essential for --cluster all baseline)
    per_cluster_metrics = {}
    for c in ["latin", "cyrillic", "greek", "arabic", "hanzi"]:
        val_mask = val_df["cluster"].values == c
        test_mask = test_df["cluster"].values == c
        c_entry = {}
        if val_mask.sum() > 0:
            c_val_f1 = float(f1_score(final_val_labels[val_mask], np.argmax(final_val_probs[val_mask], axis=1), average="macro"))
            c_val_acc = float(accuracy_score(final_val_labels[val_mask], np.argmax(final_val_probs[val_mask], axis=1)))
            c_entry["val_macro_f1"] = c_val_f1
            c_entry["val_accuracy"] = c_val_acc
        if test_mask.sum() > 0:
            c_test_f1 = float(f1_score(test_labels[test_mask], np.argmax(test_probs[test_mask], axis=1), average="macro"))
            c_test_acc = float(accuracy_score(test_labels[test_mask], np.argmax(test_probs[test_mask], axis=1)))
            c_entry["test_macro_f1"] = c_test_f1
            c_entry["test_accuracy"] = c_test_acc
        if c_entry:
            per_cluster_metrics[c] = c_entry

    # 6. Save config.json
    config_dict = {
        **vars(args),
        "label_order": LABEL_LIST_8CLASS,
        "splits_sha256": splits_sha256,
        "raw_data_sha256": raw_data_sha256,
        "git_commit": get_git_commit(),
        "package_versions": get_package_versions(),
        "lora_target_modules": LORA_TARGET_MODULES,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "device": str(device),
        "gpu_name": gpu_name,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 7. Save metrics.json
    metrics_dict = {
        "cluster": args.cluster,
        "seed": args.seed,
        "gpu_name": gpu_name,
        "epochs_trained": args.epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_val_accuracy": best_val_acc,
        "epoch_metrics": epoch_metrics,
        "total_train_seconds": round(total_train_seconds, 2),
        "overall_samples_per_sec": round(overall_samples_per_sec, 2),
        "final_val": final_val_metrics,
        "final_test": final_test_metrics,
        "per_cluster": per_cluster_metrics,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 8. Mark run DONE
    done_dict = {
        "status": "COMPLETED",
        "finished_at": datetime.utcnow().isoformat() + "Z",
        "cluster": args.cluster,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val_f1,
        "test_macro_f1": final_test_metrics["macro_f1"],
        "test_accuracy": final_test_metrics["accuracy"],
    }
    with open(done_path, "w") as f:
        json.dump(done_dict, f, indent=2)

    print(
        f"[FINISHED] Run {args.cluster} (seed {args.seed}) complete.\n"
        f"  Best Val Macro-F1: {best_val_f1:.4f} (Epoch {best_epoch})\n"
        f"  Final Test Macro-F1: {final_test_metrics['macro_f1']:.4f}\n"
        f"  Final Test Accuracy: {final_test_metrics['accuracy']:.4f}\n"
        f"  Saved to: {run_dir}"
    )

    if args.smoke and len(epoch_metrics) > 0:
        smoke_sec = epoch_metrics[0]["seconds"]
        time_per_1000 = (smoke_sec / 500) * 1000
        print(f"[SMOKE BUDGET] Time per 1,000 samples: {time_per_1000:.2f}s (~{time_per_1000/60:.2f} min)")

    if use_wandb:
        wandb.finish()

    return metrics_dict


def parse_args():
    parser = argparse.ArgumentParser(description="SFA-CA Training Pipeline (Phase C)")
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        choices=["latin", "cyrillic", "greek", "arabic", "hanzi", "all"],
        help="Cluster to train for ('all' trains monolithic baseline on union of all clusters)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for full reproducibility")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "linear"],
        help="Learning rate schedule type",
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Max gradient norm (0.0 for none)")
    parser.add_argument("--max_length", type=int, default=128, help="Max token sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.5,
        help="Supervised contrastive loss weight (within-cluster in per-cluster runs, cross-script in --cluster all)",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive temperature")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training samples using seeded class-stratified sampling",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch with 500 samples to budget real training runtime",
    )
    parser.add_argument(
        "--shuffle_labels",
        action="store_true",
        help="Shuffle TRAIN labels only for chance-level sanity check",
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        default="data/raw/multitude_v3_clean.csv",
        help="Path to raw MULTITuDE CSV",
    )
    parser.add_argument(
        "--splits_data",
        type=str,
        default="data/processed/splits_8class.csv",
        help="Path to frozen 8-class splits CSV",
    )
    parser.add_argument(
        "--allow_new_splits",
        action="store_true",
        help="Allow running if splits SHA-256 does not match frozen hash",
    )
    parser.add_argument("--out_dir", type=str, default="results/runs", help="Output directory")
    parser.add_argument("--run_tag", type=str, default="", help="Optional tag for output folder name")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB tracking if WANDB_API_KEY is present")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution (for tests)")
    parser.add_argument("--backbone", type=str, default="xlm-roberta-large", help="Transformer backbone name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_run(args)
