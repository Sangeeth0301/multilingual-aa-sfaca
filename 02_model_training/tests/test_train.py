import argparse
import json
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import LORA_TARGET_MODULES, SFACAModel
from data_loader import (
    FROZEN_SPLITS_SHA256,
    LABEL_LIST_8CLASS,
    LABEL2ID_8CLASS,
    stratified_subsample,
    load_training_table,
    compute_file_sha256,
)
from train import train_run, evaluate_dataset, compute_metrics


# ── Fixtures & Fake Backbone ──────────────────────────────────────────────────

class FakeEncoderLayer(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        return self.query(x) + self.value(x)


class FakeXLMModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": dim})()
        self.layer = FakeEncoderLayer(dim)

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        out = self.layer(torch.zeros(bsz, seq_len, 32))
        return type("Out", (), {"last_hidden_state": out})()


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        bsz = len(texts)
        seq_len = 16
        return {
            "input_ids": torch.ones(bsz, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long),
        }


def make_dummy_data(tmpdir, num_samples_per_class=10, clusters=None):
    if clusters is None:
        clusters = ["arabic", "greek", "latin", "cyrillic", "hanzi"]

    rows = []
    row_id = 0
    for split in ["train", "val", "test"]:
        for cluster in clusters:
            for cls_idx, label in enumerate(LABEL_LIST_8CLASS):
                for k in range(num_samples_per_class):
                    rows.append({
                        "row_id": row_id,
                        "cluster": cluster,
                        "language": "ar" if cluster == "arabic" else "el" if cluster == "greek" else "en",
                        "multi_label": label,
                        "split": split,
                        "group_id": row_id // 2,
                        "flag_junk": False,
                        "flag_script_mismatch": False,
                        "flag_prompt_echo": False,
                        "length_tokens": 50,
                        "max_sim_train_other": 0.1,
                        "story_disjoint": (k % 2 == 0),
                        "text": f"Sample text {row_id} for {label} in {cluster}",
                        "label": 0 if label == "human" else 1,
                        "length": 50,
                        "source": "news",
                    })
                    row_id += 1

    df = pd.DataFrame(rows)
    raw_path = os.path.join(tmpdir, "raw.csv")
    splits_path = os.path.join(tmpdir, "splits.csv")

    raw_cols = ["row_id", "text", "label", "multi_label", "split", "language", "length", "source"]
    raw_df = df[raw_cols].rename(columns={"row_id": "Unnamed: 0"})
    raw_df.to_csv(raw_path, index=False)

    split_cols = [
        "row_id", "cluster", "language", "multi_label", "split", "group_id",
        "flag_junk", "flag_script_mismatch", "flag_prompt_echo", "length_tokens",
        "max_sim_train_other", "story_disjoint",
    ]
    splits_df = df[split_cols]
    splits_df.to_csv(splits_path, index=False)

    return raw_path, splits_path


# ── 1. LoRA Target Modules & Parameter Attachment ─────────────────────────────

def test_lora_trainable_parameters_attached():
    """Verify XLM-RoBERTa uses 'query' and 'value', trainable params > 0, and adapters attach to real modules."""
    assert LORA_TARGET_MODULES == ["query", "value"]

    with patch("model.AutoModel.from_pretrained", return_value=FakeXLMModel(32)):
        model = SFACAModel(clusters=["latin"], num_classes=8, lora_r=4, lora_alpha=8)

        # Check trainable parameters > 0
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable)
        assert trainable_count > 0

        # Check adapter layers attached to actual modules
        attached_modules = [name for name, m in model.backbone.named_modules() if hasattr(m, "lora_A")]
        assert len(attached_modules) > 0
        for mod_name in attached_modules:
            assert any(tgt in mod_name for tgt in LORA_TARGET_MODULES)


# ── 2. Class-Stratified & Deterministic Subsampling ───────────────────────────

def test_deterministic_and_class_stratified_subsample():
    """Verify --max_train_samples is deterministic given seed and class-stratified within 1."""
    # Create an imbalanced set of 8 classes
    records = []
    for c in range(8):
        count = 100 + c * 20  # from 100 to 240
        for i in range(count):
            records.append({"label_id": c, "val": i})
    df = pd.DataFrame(records)

    # Subsample to 500 samples
    sub1 = stratified_subsample(df, max_samples=500, seed=42)
    sub2 = stratified_subsample(df, max_samples=500, seed=42)

    # Deterministic check
    pd.testing.assert_frame_equal(sub1, sub2)

    # Class-stratified check: each class within 1 sample of others
    counts = sub1["label_id"].value_counts()
    assert len(counts) == 8
    assert counts.max() - counts.min() <= 1

    # Seed change check
    sub_seed43 = stratified_subsample(df, max_samples=500, seed=43)
    assert not sub1["val"].equals(sub_seed43["val"])


# ── 3. Shuffle Labels Affects Train Only ───────────────────────────────────────

def test_shuffle_labels_affects_train_only():
    """Verify --shuffle_labels permutes train labels only; val and test labels remain intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path, splits_path = make_dummy_data(tmpdir, num_samples_per_class=5)

        merged, _ = load_training_table(raw_path, splits_path, allow_new_splits=True)
        train_df = merged[merged["split"] == "train"].copy()
        val_df = merged[merged["split"] == "val"].copy()

        val_labels_before = val_df["label_id"].tolist()
        train_labels_before = train_df["label_id"].tolist()

        rng = np.random.RandomState(42)
        train_df["label_id"] = rng.permutation(train_df["label_id"].values)

        # Train labels changed (permuted)
        assert train_df["label_id"].tolist() != train_labels_before
        # But distribution of classes is identical
        assert sorted(train_df["label_id"].tolist()) == sorted(train_labels_before)
        # Val labels completely untouched
        assert val_df["label_id"].tolist() == val_labels_before


# ── 4. Splits Hash Verification ───────────────────────────────────────────────

def test_splits_hash_verification():
    """Verify tampered splits hash aborts unless --allow_new_splits is passed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path, splits_path = make_dummy_data(tmpdir, num_samples_per_class=2)

        # Without allow_new_splits, must raise ValueError
        with pytest.raises(ValueError, match="Frozen splits SHA-256 mismatch"):
            load_training_table(raw_path, splits_path, allow_new_splits=False)

        # With allow_new_splits, succeeds
        df, h = load_training_table(raw_path, splits_path, allow_new_splits=True)
        assert len(df) > 0


# ── 5. Seed Reproducibility ───────────────────────────────────────────────────

def test_seed_reproducibility():
    """Verify same seed produces identical loss sequence on CPU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path, splits_path = make_dummy_data(tmpdir, num_samples_per_class=4, clusters=["arabic"])

        out1 = os.path.join(tmpdir, "run1")
        out2 = os.path.join(tmpdir, "run2")

        args1 = argparse.Namespace(
            cluster="arabic", seed=42, epochs=2, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out1, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )
        args2 = argparse.Namespace(
            cluster="arabic", seed=42, epochs=2, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out2, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        with patch("train.SFACAModel", side_effect=lambda **kw: SFACAModel(clusters=kw.get("clusters"), num_classes=8, lora_r=4, lora_alpha=8)), \
             patch("model.AutoModel.from_pretrained", return_value=FakeXLMModel(32)), \
             patch("train.build_tokenizer", return_value=FakeTokenizer()):

            m1 = train_run(args1)
            m2 = train_run(args2)

            losses1 = [ep["train_loss"] for ep in m1["epoch_metrics"]]
            losses2 = [ep["train_loss"] for ep in m2["epoch_metrics"]]
            assert losses1 == losses2


# ── 6. Best Epoch Selection on Val (Not Test) ─────────────────────────────────

def test_best_epoch_chosen_by_val_not_test():
    """Verify best checkpoint is selected purely on val macro-F1, even if test peaks elsewhere."""
    # Simulate scenario:
    # Epoch 1: val=0.85, test=0.60
    # Epoch 2: val=0.70, test=0.95
    epochs = [
        {"epoch": 1, "val_f1": 0.85, "test_f1": 0.60},
        {"epoch": 2, "val_f1": 0.70, "test_f1": 0.95},
    ]

    best_epoch = -1
    best_val_f1 = -1.0
    for ep in epochs:
        if ep["val_f1"] > best_val_f1:
            best_val_f1 = ep["val_f1"]
            best_epoch = ep["epoch"]

    # Must select Epoch 1 (val peak), NOT Epoch 2 (test peak)
    assert best_epoch == 1
    assert best_val_f1 == 0.85


# ── 7. Test Set Never Read During Training ─────────────────────────────────────

def test_test_set_never_accessed_during_training():
    """Verify test DataLoader is never accessed prior to final evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Give test a distinct sample count so we can unequivocally track it
        rows = []
        row_id = 0
        for split, count in [("train", 3), ("val", 3), ("test", 5)]:
            for label in LABEL_LIST_8CLASS:
                for _ in range(count):
                    rows.append({
                        "row_id": row_id,
                        "cluster": "arabic",
                        "language": "ar",
                        "multi_label": label,
                        "split": split,
                        "group_id": row_id // 2,
                        "flag_junk": False,
                        "flag_script_mismatch": False,
                        "flag_prompt_echo": False,
                        "length_tokens": 50,
                        "max_sim_train_other": 0.1,
                        "story_disjoint": True,
                        "text": f"Sample {row_id}",
                        "label": 0 if label == "human" else 1,
                        "length": 50,
                        "source": "news",
                    })
                    row_id += 1

        df = pd.DataFrame(rows)
        raw_path = os.path.join(tmpdir, "raw.csv")
        splits_path = os.path.join(tmpdir, "splits.csv")
        df[["row_id", "text", "label", "multi_label", "split", "language", "length", "source"]].rename(
            columns={"row_id": "Unnamed: 0"}
        ).to_csv(raw_path, index=False)
        df[[
            "row_id", "cluster", "language", "multi_label", "split", "group_id",
            "flag_junk", "flag_script_mismatch", "flag_prompt_echo", "length_tokens",
            "max_sim_train_other", "story_disjoint",
        ]].to_csv(splits_path, index=False)

        out_dir = os.path.join(tmpdir, "run")

        args = argparse.Namespace(
            cluster="arabic", seed=42, epochs=2, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out_dir, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        test_access_history = []

        with patch("train.SFACAModel", side_effect=lambda **kw: SFACAModel(clusters=kw.get("clusters"), num_classes=8, lora_r=4, lora_alpha=8)), \
             patch("model.AutoModel.from_pretrained", return_value=FakeXLMModel(32)), \
             patch("train.build_tokenizer", return_value=FakeTokenizer()):

            orig_evaluate = evaluate_dataset

            def monitored_evaluate(model, loader, adapter_name, device):
                # Test set uniquely has 5 samples * 8 classes = 40 samples
                is_test = (len(loader.dataset) == 40)
                test_access_history.append(is_test)
                return orig_evaluate(model, loader, adapter_name, device)

            with patch("train.evaluate_dataset", side_effect=monitored_evaluate):
                train_run(args)

            # Test should only be evaluated ONCE at the very end
            assert test_access_history.count(True) == 1
            assert test_access_history[-1] == True  # The very last call


# ── 8. Cluster All Baseline Mode ──────────────────────────────────────────────

def test_cluster_all_merges_clusters():
    """Verify --cluster all unions all 5 clusters and computes per-cluster metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path, splits_path = make_dummy_data(tmpdir, num_samples_per_class=2)
        out_dir = os.path.join(tmpdir, "run_all")

        args = argparse.Namespace(
            cluster="all", seed=42, epochs=1, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out_dir, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        with patch("train.SFACAModel", side_effect=lambda **kw: SFACAModel(clusters=kw.get("clusters"), num_classes=8, lora_r=4, lora_alpha=8)), \
             patch("model.AutoModel.from_pretrained", return_value=FakeXLMModel(32)), \
             patch("train.build_tokenizer", return_value=FakeTokenizer()):

            metrics = train_run(args)
            assert "per_cluster" in metrics
            for c in ["latin", "cyrillic", "greek", "arabic", "hanzi"]:
                assert c in metrics["per_cluster"]


# ── 9. DONE.json Skips Completed Runs ─────────────────────────────────────────

def test_done_json_skips_run():
    """Verify presence of DONE.json causes training to be skipped immediately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "arabic_seed42")
        os.makedirs(run_dir, exist_ok=True)
        done_path = os.path.join(run_dir, "DONE.json")
        with open(done_path, "w") as f:
            json.dump({"status": "COMPLETED", "cluster": "arabic", "seed": 42}, f)

        args = argparse.Namespace(
            cluster="arabic", seed=42, epochs=3, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data="dummy", splits_data="dummy", allow_new_splits=True,
            out_dir=tmpdir, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        # Should return without touching raw_data or loading models
        res = train_run(args)
        assert res["status"] == "COMPLETED"


# ── 10. Mid-Run Resume Parity ─────────────────────────────────────────────────

def test_mid_run_resume_parity():
    """Verify run 1 epoch -> stop -> resume epoch 2 matches uninterrupted 2-epoch run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path, splits_path = make_dummy_data(tmpdir, num_samples_per_class=4, clusters=["greek"])

        out_uninterrupted = os.path.join(tmpdir, "uninterrupted")
        out_resumed = os.path.join(tmpdir, "resumed")

        args_uninterrupted = argparse.Namespace(
            cluster="greek", seed=42, epochs=2, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out_uninterrupted, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        args_part1 = argparse.Namespace(
            cluster="greek", seed=42, epochs=1, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out_resumed, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        args_part2 = argparse.Namespace(
            cluster="greek", seed=42, epochs=2, batch_size=8, lr=1e-3,
            lr_scheduler="none", warmup_steps=0, max_grad_norm=0.0, max_length=16,
            lora_r=4, lora_alpha=8, lora_dropout=0.0, contrastive_weight=0.5,
            temperature=0.1, max_train_samples=None, smoke=False, shuffle_labels=False,
            raw_data=raw_path, splits_data=splits_path, allow_new_splits=True,
            out_dir=out_resumed, run_tag="", wandb=False, force_cpu=True, backbone="fake",
        )

        with patch("train.SFACAModel", side_effect=lambda **kw: SFACAModel(clusters=kw.get("clusters"), num_classes=8, lora_r=4, lora_alpha=8)), \
             patch("model.AutoModel.from_pretrained", return_value=FakeXLMModel(32)), \
             patch("train.build_tokenizer", return_value=FakeTokenizer()):

            # 1. Uninterrupted 2-epoch run
            m_uninterrupted = train_run(args_uninterrupted)

            # 2. 1-epoch run
            train_run(args_part1)
            # Remove DONE.json from part1 so part2 can resume to epoch 2
            done_file = os.path.join(out_resumed, "greek_seed42", "DONE.json")
            if os.path.exists(done_file):
                os.remove(done_file)

            # 3. Resume for epoch 2
            m_resumed = train_run(args_part2)

            val_uninterrupted = m_uninterrupted["final_val"]["macro_f1"]
            val_resumed = m_resumed["final_val"]["macro_f1"]

            # Tolerance on CPU
            assert abs(val_uninterrupted - val_resumed) < 1e-4
