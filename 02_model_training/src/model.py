"""
model.py
--------
SFA-CA model definition: a frozen multilingual backbone (XLM-RoBERTa-large)
with one LoRA adapter per script/family cluster, plus a shared classification
head on top.

Key idea: instead of a single fine-tuned model for all languages, we keep
the backbone frozen and train small, swappable LoRA adapters -- one per
cluster (Latin, Cyrillic, Greek, Arabic, Hanzi, ...). At train/inference
time, we activate the adapter matching the input's cluster.

Uses Hugging Face `peft`, which implements LoRA for you -- you do not
need to hand-write the low-rank matrix math.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from clusters import all_clusters


BACKBONE_NAME = "xlm-roberta-large"  # same backbone as base paper's OTBDetector
NUM_CLASSES = 8                       # human + 7 generators (8-class task)
LORA_TARGET_MODULES = ["query", "value"]  # XLM-RoBERTa attention projections


class SFACAModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        num_classes: int = NUM_CLASSES,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules=None,
        clusters=None,
    ):
        super().__init__()
        self.clusters = clusters or all_clusters()
        if target_modules is None:
            target_modules = LORA_TARGET_MODULES

        # 1. Load the shared multilingual backbone and freeze it entirely.
        base_model = AutoModel.from_pretrained(backbone_name)
        for param in base_model.parameters():
            param.requires_grad = False

        # 2. Wrap it with peft so we can attach multiple named LoRA adapters,
        #    one per script/family cluster.
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,  # attention projections
            bias="none",
            use_dora=False,  # Prevents PEFT from probing torchao dispatcher
        )
        # First adapter is added at wrap time; we add the rest by name below.
        self.backbone = get_peft_model(base_model, lora_config,
                                        adapter_name=self.clusters[0])
        for cluster_name in self.clusters[1:]:
            self.backbone.add_adapter(cluster_name, lora_config)

        hidden_size = base_model.config.hidden_size

        # 3. Shared classification head (same for every cluster).
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def set_active_cluster(self, cluster_name: str):
        """Swap which LoRA adapter is currently active before a forward pass."""
        self.backbone.set_adapter(cluster_name)

    def forward(self, input_ids, attention_mask, cluster_name: str):
        self.set_active_cluster(cluster_name)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS]-equivalent (first token) pooled representation.
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits

    def get_embedding(self, input_ids, attention_mask, cluster_name: str):
        """Return the pooled representation only (used by the contrastive loss)."""
        self.set_active_cluster(cluster_name)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


def build_tokenizer(backbone_name: str = BACKBONE_NAME):
    return AutoTokenizer.from_pretrained(backbone_name)


if __name__ == "__main__":
    # Quick smoke test (downloads xlm-roberta-large the first time you run this).
    tok = build_tokenizer()
    model = SFACAModel(clusters=["latin", "cyrillic"])

    sample = tok(["Hello world", "Privet mir"], padding=True, return_tensors="pt")
    logits = model(sample["input_ids"], sample["attention_mask"], cluster_name="latin")
    print("Logits shape:", logits.shape)  # expect: [2, 8]
