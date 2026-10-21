"""
push_to_hub.py
--------------
Hugging Face Model Hub Integration for SFA-CA.

Publishes trained LoRA script-family adapters to the Hugging Face Hub so anyone can
load them using `peft.PeftModel.from_pretrained(...)`.

Usage:
    python 03_inference_evaluation/src/push_to_hub.py --repo_prefix sandeepsakthi0301/sfaca --adapters_dir results/adapters
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import argparse
import os
import torch
from huggingface_hub import HfApi, create_repo

CLUSTERS = ["latin", "cyrillic", "greek", "arabic", "hanzi"]

def push_adapters(repo_prefix: str, adapters_dir: str, token: str = None):
    api = HfApi(token=token)

    for cluster in CLUSTERS:
        cluster_path = os.path.join(adapters_dir, cluster)
        if not os.path.exists(cluster_path):
            print(f"Skipping '{cluster}': directory not found at {cluster_path}")
            continue

        repo_id = f"{repo_prefix}-{cluster}-lora"
        print(f"\nPublishing adapter '{cluster}' to Hugging Face Hub: {repo_id}...")

        try:
            create_repo(repo_id=repo_id, exist_ok=True, token=token)
            
            # Create a simple model card (README.md) for HF Hub
            model_card_content = f"""---
language:
- multilingual
library_name: peft
pipeline_tag: text-classification
tags:
- authorship-attribution
- sfaca
- lora
- script-family-{cluster}
---

# SFA-CA LoRA Adapter: {cluster.upper()} Script Family

This is a **Script-Family-Aware Contrastive Adaptation (SFA-CA)** LoRA adapter for **{cluster.capitalize()}** script family languages.
Trained on top of frozen `xlm-roberta-large` for Multilingual Authorship Attribution (Human vs. AI Generators).

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

base_model = AutoModel.from_pretrained("xlm-roberta-large")
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
```
"""
            model_card_path = os.path.join(cluster_path, "README.md")
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)

            api.upload_folder(
                folder_path=cluster_path,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"✅ Successfully published {repo_id} to Hugging Face Hub!")
        except Exception as e:
            print(f"Error publishing {repo_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push SFA-CA Adapters to Hugging Face Hub")
    parser.add_argument("--repo_prefix", type=str, default="sandeepsakthi0301/sfaca", help="Prefix for HF repo names")
    parser.add_argument("--adapters_dir", type=str, default="results/adapters", help="Local directory containing adapters")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    push_adapters(args.repo_prefix, args.adapters_dir, args.token)
