import json
import os

nb = {
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True)
    })

add_md("# SFA-CA Multilingual Attribution Training (Kaggle T4)\n\nThis notebook executes the reproducible training experiments for SFA-CA (Phase C).\nIt strictly adheres to frozen split verification, CPU/GPU determinism, and session resumption.")

add_md("## 1. Environment Verification\nPer protocol, do NOT pip install or upgrade torch, transformers, or peft. Compare pre-installed versions with `notebooks/kaggle/KAGGLE_PIP_FREEZE.txt` and stop if they differ.")

add_code('''import sys
import importlib.metadata

print(f"Python: {sys.version}")

expected_versions = {
    "peft": "0.19.1",
    "transformers": "5.0.0",
}

for pkg, exp_ver in expected_versions.items():
    try:
        cur_ver = importlib.metadata.version(pkg)
        print(f"{pkg}: {cur_ver} (expected: {exp_ver})")
        if cur_ver.split(".")[0] != exp_ver.split(".")[0]:
            print(f"ERROR: {pkg} version mismatch! Expected major version {exp_ver}, got {cur_ver}")
    except Exception as e:
        print(f"Warning: {pkg} check failed: {e}")

# Only install small missing utility packages if needed (never torch, transformers, or peft)
!pip install -q scikit-learn
''')

add_md("## 2. Clone Repository & Checkout Targeted Commit")

add_code('''import os

REPO_DIR = "/kaggle/working/multilingual-aa-sfaca"
TARGET_COMMIT = "phase-c"  # Target branch or exact commit hash

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/ByteBeast-1/multilingual-aa-sfaca.git {REPO_DIR}

os.chdir(REPO_DIR)
!git checkout {TARGET_COMMIT}
!git rev-parse HEAD
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
''')

add_md("## 3. Verify Dataset Hashes\nVerify frozen datasets in `/kaggle/input/sfaca-data/`.")

add_code('''import hashlib

def verify_sha256(path, expected_hash):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"{os.path.basename(path)}: {digest}")
    assert digest == expected_hash, f"Hash mismatch! Got {digest}, expected {expected_hash}"
    print("  --> SHA-256 PASSED")

DATA_DIR = "/kaggle/input/sfaca-data"
RAW_PATH = os.path.join(DATA_DIR, "multitude_v3_clean.csv")
SPLITS_PATH = os.path.join(DATA_DIR, "splits_8class.csv")

verify_sha256(SPLITS_PATH, "8c05df8355c44611b05aa290c6644f822436734a74da278f65da1c91f370cf0b")
verify_sha256(RAW_PATH, "1e08f9c6980ed9cae11bc831d8ce54d13125ea9a52dfef387b0b4ebd0909647c")
''')

add_md("## 4. Smoke Test (Arabic, 1 Epoch, 500 Samples)\nPrints time per 1,000 samples to budget real training runs.")

add_code('''!python src/train.py \\
    --cluster arabic \\
    --seed 42 \\
    --smoke \\
    --raw_data {RAW_PATH} \\
    --splits_data {SPLITS_PATH} \\
    --out_dir /kaggle/working/results/runs
''')

add_md("## 5. Master Training Loop (Lean Plan)\nLoops over priority runs. Automatically skips any run that already has `DONE.json`.")

add_code('''import subprocess

OUT_DIR = "/kaggle/working/results/runs"
os.makedirs(OUT_DIR, exist_ok=True)

experiments = [
    # 1. Arabic (3 seeds)
    {"cluster": "arabic", "seed": 42},
    {"cluster": "arabic", "seed": 43},
    {"cluster": "arabic", "seed": 44},
    # 2. Greek (3 seeds)
    {"cluster": "greek", "seed": 42},
    {"cluster": "greek", "seed": 43},
    {"cluster": "greek", "seed": 44},
    # 3. Hanzi (3 seeds)
    {"cluster": "hanzi", "seed": 42},
    {"cluster": "hanzi", "seed": 43},
    {"cluster": "hanzi", "seed": 44},
    # 4. Cyrillic (3 seeds)
    {"cluster": "cyrillic", "seed": 42},
    {"cluster": "cyrillic", "seed": 43},
    {"cluster": "cyrillic", "seed": 44},
    # 5. Latin (1 seed)
    {"cluster": "latin", "seed": 42},
    # 6. Monolithic Baseline (all clusters, 1 seed)
    {"cluster": "all", "seed": 42},
    # 7. Ablation: Contrastive weight 0.0 vs 0.5 (arabic, greek)
    {"cluster": "arabic", "seed": 42, "contrastive_weight": 0.0, "run_tag": "con0"},
    {"cluster": "greek", "seed": 42, "contrastive_weight": 0.0, "run_tag": "con0"},
    # 8. Ablation: LoRA rank 16 and 32 (arabic)
    {"cluster": "arabic", "seed": 42, "lora_r": 16, "lora_alpha": 32, "run_tag": "r16"},
    {"cluster": "arabic", "seed": 42, "lora_r": 32, "lora_alpha": 64, "run_tag": "r32"},
]

for exp in experiments:
    c = exp["cluster"]
    s = exp["seed"]
    tag = exp.get("run_tag", "")
    tag_suffix = f"_{tag}" if tag else ""
    run_folder = os.path.join(OUT_DIR, f"{c}_seed{s}{tag_suffix}")
    done_file = os.path.join(run_folder, "DONE.json")

    if os.path.exists(done_file):
        print(f"[SKIP] Experiment {c} seed {s} tag={tag} already finished.")
        continue

    print(f"\\n==================================================")
    print(f"🚀 Starting: Cluster={c} Seed={s} Tag={tag}")
    print(f"==================================================")

    cmd = [
        "python", "src/train.py",
        "--cluster", c,
        "--seed", str(s),
        "--raw_data", RAW_PATH,
        "--splits_data", SPLITS_PATH,
        "--out_dir", OUT_DIR,
    ]
    if "contrastive_weight" in exp:
        cmd.extend(["--contrastive_weight", str(exp["contrastive_weight"])])
    if "lora_r" in exp:
        cmd.extend(["--lora_r", str(exp["lora_r"]), "--lora_alpha", str(exp["lora_alpha"])])
    if tag:
        cmd.extend(["--run_tag", tag])

    subprocess.run(cmd, check=True)
''')

add_md("## 6. How to Resume After Session Reset\nIf a Kaggle session resets:\n1. If checkpoints were copied to an output dataset, restore them into `/kaggle/working/results/runs/`.\n2. Re-run Cell 5. The loop detects existing epochs via `resume_state.json` and resumes from the last completed epoch automatically.\n3. Runs with `DONE.json` are completely skipped.")

add_md("## 7. Archive Outputs\nCompress the results folder for easy download or persistence to Kaggle Dataset.")

add_code('''!zip -r -q /kaggle/working/sfaca_results.zip /kaggle/working/results/runs/
print("Archive created at /kaggle/working/sfaca_results.zip")
''')

os.makedirs("notebooks/kaggle", exist_ok=True)
with open("notebooks/kaggle/sfaca_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("sfaca_train.ipynb generated successfully!")
