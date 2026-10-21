"""
data_loader.py
--------------
Loads MULTITuDE v3 data and prepares it for SFA-CA training/evaluation.

Expected input format: a CSV (or set of per-language CSVs) with at least
these columns:
    text        -> the raw text sample
    label       -> generator class, one of:
                   "human", "mistral", "opt", "eagle", "vicuna",
                   "llama2", "aya", "gpt35"
    lang        -> ISO 639-1 language code (e.g. "en", "ru", "zh")

Adjust COLUMN_MAP / load paths to match however you actually download
MULTITuDE v3 from Zenodo -- the base paper's repo
(github.com/MLNTeam-Unical/Multilingual-MGT-AA) shows their exact schema.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import hashlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from clusters import get_cluster, all_clusters, BASE_PAPER_18_LANGUAGES


LABEL2ID = {
    "human": 0,
    "Mistral-7B-Instruct-v0.2": 1,
    "opt-iml-max-30b": 2,
    "v5-Eagle-7B-HF": 3,
    "vicuna-13b": 4,
    "Llama-2-70b-chat-hf": 5,
    "aya-101": 6,
    "gpt-3.5-turbo-0125": 7,
    "Gemini-2.5-Flash": 8,
    "Claude-Haiku-4.5": 9,
    # Legacy aliases (backward compat with old CSV labels)
    "Gemini-1.5-Flash": 8,
    "gemini": 8,
    "gemini-flash": 8,
    "gemini-3.6-flash": 8,
    "Claude-3.5-Sonnet": 9,
    "claude": 9,
    "claude-3.5-sonnet": 9,
}
ID2LABEL = {
    0: "human",
    1: "Mistral-7B-Instruct-v0.2",
    2: "opt-iml-max-30b",
    3: "v5-Eagle-7B-HF",
    4: "vicuna-13b",
    5: "Llama-2-70b-chat-hf",
    6: "aya-101",
    7: "gpt-3.5-turbo-0125",
    8: "Gemini-2.5-Flash",
    9: "Claude-Haiku-4.5",
}


def load_multitude_csv(path: str, restrict_to_base_paper_18: bool = True) -> pd.DataFrame:
    """
    Load the raw MULTITuDE v3 dataframe and prepare it for SFA-CA.

    The real file uses these column names (confirmed from the actual
    downloaded multitude_v3_clean.csv):
        text         -> the raw text sample
        label        -> BINARY 0=human, 1=machine (not used for attribution)
        multi_label  -> the actual class we need: "human" or a specific
                        generator name, e.g. "gpt-3.5-turbo-0125"
        split        -> "train" or "test" (already provided by the dataset)
        language     -> ISO 639-1 language code (21 total in the raw file)

    restrict_to_base_paper_18: if True, drops the 3 extra languages
        (ca, ga, gd) that are in the raw file but were NOT part of the
        base paper's 18-language selection, so your results stay directly
        comparable to their reported numbers (Table 1, Table 2, etc.).
    """
    df = pd.read_csv(path)

    if restrict_to_base_paper_18:
        df = df[df["language"].isin(BASE_PAPER_18_LANGUAGES)].reset_index(drop=True)

    df["cluster"] = df["language"].apply(get_cluster)
    df["label_id"] = df["multi_label"].map(LABEL2ID)

    # Sanity check: flag any rows whose multi_label didn't match LABEL2ID
    # (would show up as NaN in label_id) instead of silently training on
    # broken labels.
    unmapped = df["label_id"].isna().sum()
    if unmapped > 0:
        unknown_values = df.loc[df["label_id"].isna(), "multi_label"].unique()
        raise ValueError(
            f"{unmapped} rows have a multi_label value not found in "
            f"LABEL2ID: {unknown_values}. Update LABEL2ID in data_loader.py."
        )

    return df


def filter_by_clusters(df: pd.DataFrame, clusters: list) -> pd.DataFrame:
    """Keep only rows whose language cluster is in `clusters`."""
    return df[df["cluster"].isin(clusters)].reset_index(drop=True)


def filter_by_languages(df: pd.DataFrame, langs: list) -> pd.DataFrame:
    """Keep only rows whose language code is in `langs`."""
    return df[df["language"].isin(langs)].reset_index(drop=True)


FROZEN_SPLITS_SHA256 = "8c05df8355c44611b05aa290c6644f822436734a74da278f65da1c91f370cf0b"

LABEL_LIST_8CLASS = [
    "human",
    "Llama-2-70b-chat-hf",
    "Mistral-7B-Instruct-v0.2",
    "aya-101",
    "gpt-3.5-turbo-0125",
    "opt-iml-max-30b",
    "v5-Eagle-7B-HF",
    "vicuna-13b",
]
LABEL2ID_8CLASS = {lbl: i for i, lbl in enumerate(LABEL_LIST_8CLASS)}
ID2LABEL_8CLASS = {i: lbl for i, lbl in enumerate(LABEL_LIST_8CLASS)}


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_training_table(
    raw_path: str = "data/raw/multitude_v3_clean.csv",
    splits_path: str = "data/processed/splits_8class.csv",
    allow_new_splits: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Loads raw CSV and splits CSV joined on row_id (verifying row_id equals Unnamed: 0).
    Verifies SHA-256 of splits_path against FROZEN_SPLITS_SHA256 unless allow_new_splits=True.
    Returns (merged_df, splits_sha256).
    """
    splits_hash = compute_file_sha256(splits_path)
    if splits_hash != FROZEN_SPLITS_SHA256 and not allow_new_splits:
        raise ValueError(
            f"Frozen splits SHA-256 mismatch!\nExpected: {FROZEN_SPLITS_SHA256}\nFound:    {splits_hash}\n"
            f"Pass --allow_new_splits to override."
        )

    splits_df = pd.read_csv(splits_path)
    raw_df = pd.read_csv(raw_path)

    if "Unnamed: 0" not in raw_df.columns:
        raise ValueError("raw CSV must contain 'Unnamed: 0' column for row indexing.")
    if "row_id" not in splits_df.columns:
        raise ValueError("splits CSV must contain 'row_id' column.")

    raw_df = raw_df.rename(columns={"Unnamed: 0": "row_id"})

    cols_to_keep = [c for c in raw_df.columns if c not in splits_df.columns or c == "row_id"]
    merged_df = pd.merge(splits_df, raw_df[cols_to_keep], on="row_id", how="inner")

    if len(merged_df) != len(splits_df):
        raise ValueError(f"Join mismatch: splits has {len(splits_df)} rows, but joined result has {len(merged_df)} rows.")

    merged_df["label_id"] = merged_df["multi_label"].map(LABEL2ID_8CLASS)
    if merged_df["label_id"].isna().any():
        unmapped = merged_df.loc[merged_df["label_id"].isna(), "multi_label"].unique()
        raise ValueError(f"Unknown multi_label values for 8-class setup: {unmapped}")
    merged_df["label_id"] = merged_df["label_id"].astype(int)

    return merged_df, splits_hash


def stratified_subsample(df: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Deterministically subsample df to at most max_samples such that:
    - Samples are class-stratified (each class count is within 1 of others when data allows).
    - Result is deterministic given the seed.
    """
    if max_samples is None or len(df) <= max_samples:
        return df.copy()

    rng = np.random.RandomState(seed)
    classes = sorted(df["label_id"].unique())
    class_indices = {c: df.index[df["label_id"] == c].tolist() for c in classes}
    class_sizes = {c: len(idx) for c, idx in class_indices.items()}

    shuffled_indices = {}
    for c in classes:
        idx = list(class_indices[c])
        rng.shuffle(idx)
        shuffled_indices[c] = idx

    allocated = {c: 0 for c in classes}
    remaining_budget = max_samples

    while remaining_budget > 0:
        available_classes = [c for c in classes if allocated[c] < class_sizes[c]]
        if not available_classes:
            break
        for c in available_classes:
            if remaining_budget == 0:
                break
            allocated[c] += 1
            remaining_budget -= 1

    selected_indices = []
    for c in classes:
        selected_indices.extend(shuffled_indices[c][:allocated[c]])

    selected_indices = sorted(selected_indices)
    return df.loc[selected_indices].copy().reset_index(drop=True)


def filter_by_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Keep only rows matching the dataset split ('train', 'val', or 'test')."""
    assert split_name in ("train", "val", "test"), f"split_name must be 'train', 'val', or 'test', got '{split_name}'"
    return df[df["split"] == split_name].reset_index(drop=True)


class AttributionDataset(Dataset):
    """
    A thin torch Dataset wrapper around a dataframe, ready to
    be tokenized on the fly in the training loop.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.clusters = df["cluster"].tolist()
        self.row_ids = df["row_id"].tolist() if "row_id" in df.columns else list(range(len(df)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = self.labels[idx]
        item["cluster"] = self.clusters[idx]
        item["row_id"] = self.row_ids[idx]
        return item


if __name__ == "__main__":
    # Example usage once you have the real CSV downloaded:
    # df = load_multitude_csv("data/raw/multitude_v3.csv")
    # print(df["cluster"].value_counts())
    print("Defined clusters:", all_clusters())
    print("Update the path in load_multitude_csv(...) once your data is downloaded.")
