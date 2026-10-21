---
title: SFA-CA AI Text Attribution
emoji: 🔎
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

Detects whether text is human- or AI-written, which passages look AI-generated, and which of 7 generators
(GPT-3.5, Llama-2, Mistral-7B, Vicuna, OPT-IML, Eagle-7B, Aya-101) is the closest match.
Script-family LoRA adapters on XLM-RoBERTa-large (Latin, Cyrillic, Greek, Arabic, Chinese).
Output is a statistical likelihood, not proof of authorship.
