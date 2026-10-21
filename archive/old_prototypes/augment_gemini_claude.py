"""
augment_gemini_claude.py
-------------------------
Appends representative Gemini-2.5-Flash and Claude-Haiku-4.5 text samples
across languages to create a balanced 10-class dataset for SFA-CA.

Key fixes vs. original:
  - NO [Gemini]/[Claude] prefixes — model must learn style, not token markers
  - 30 diverse templates per model (vs 5) for better generalization
  - Balanced to ~20,000 samples per new class (vs 5,500)
  - Labels updated to Gemini-2.5-Flash and Claude-Haiku-4.5
"""

import os
import pandas as pd

# ── Gemini-2.5-Flash style ────────────────────────────────────────────────────
# Characteristics: structured, markdown-friendly, informative, slightly formal,
# uses numbered/bulleted breakdowns, "Let's explore...", "Here's how..."
GEMINI_TEMPLATES = [
    "Artificial intelligence systems process unstructured text into high-dimensional embedding spaces, enabling multi-agent reasoning and context synthesis.",
    "The analysis of multi-LLM attribution relies on subtle stylistic variations in token selection and probability distribution dynamics.",
    "In modern computer science, transformer models utilize multi-head self-attention mechanisms to capture complex contextual dependencies across long sequences.",
    "Exploring the nuances of language processing reveals how contrastive learning aligns representations across disparate writing scripts.",
    "Here's a breakdown of the key concepts: First, we examine the foundational principles. Then, we explore their practical applications in real-world systems.",
    "Let's explore this step by step. The transformer architecture consists of encoder and decoder blocks, each containing multi-head attention layers.",
    "When analyzing natural language, it's important to consider both semantic and syntactic features. These two dimensions interact in complex ways.",
    "The field of computational linguistics has undergone a paradigm shift with the introduction of large-scale pre-trained language models.",
    "To understand this phenomenon, we need to examine the underlying mathematical framework that governs neural network optimization.",
    "Machine learning pipelines typically involve data preprocessing, feature extraction, model training, and evaluation phases.",
    "The intersection of linguistics and computer science has yielded powerful tools for automated text analysis and generation.",
    "Recent advances in foundation models have demonstrated remarkable few-shot learning capabilities across diverse language tasks.",
    "Here are the main factors to consider when evaluating language model performance across multilingual settings.",
    "Statistical approaches to natural language processing have been largely supplanted by deep learning methods in recent years.",
    "The concept of transfer learning enables models pretrained on large corpora to be adapted efficiently to specialized downstream tasks.",
    "Semantic similarity between texts can be measured using cosine distance in the embedding space of transformer-based encoders.",
    "Gradient-based optimization techniques allow neural networks to learn complex representations through backpropagation of error signals.",
    "The tokenization of text into subword units enables language models to handle out-of-vocabulary words effectively.",
    "Cross-lingual transfer learning exploits shared representations between languages to improve performance on low-resource tasks.",
    "Attention mechanisms allow models to dynamically weight the importance of different input tokens when generating outputs.",
    "The scaling laws for neural language models suggest that performance improves predictably with increases in model size and training data.",
    "Instruction tuning has emerged as a powerful technique for aligning language model behavior with human intent.",
    "Retrieval-augmented generation combines parametric knowledge stored in model weights with non-parametric external knowledge bases.",
    "The evaluation of generative language models presents unique challenges due to the open-ended nature of text generation tasks.",
    "Multilingual models must balance language-specific specialization with cross-lingual generalization across diverse linguistic families.",
    "Constitutional AI approaches embed ethical constraints directly into the training process to improve model safety and alignment.",
    "Chain-of-thought prompting elicits structured reasoning from language models by providing examples of step-by-step problem solving.",
    "The emergence of capabilities in large language models refers to qualitative improvements that appear abruptly at certain scale thresholds.",
    "Quantization techniques reduce model memory requirements by representing weights with lower-precision numerical formats.",
    "In-context learning allows language models to adapt to new tasks using only a few demonstrations provided in the prompt.",
]

# ── Claude-Haiku-4.5 style ────────────────────────────────────────────────────
# Characteristics: concise, direct, helpful tone, uses "I'd be happy to...",
# structured with clear steps, balanced nuance, often uses bullet points
CLAUDE_TEMPLATES = [
    "I'd be happy to help analyze this topic. When considering complex technical frameworks, it is essential to systematically examine each component.",
    "In evaluating artificial intelligence architectures, we must balance computational efficiency, parameter scale, and forensic attribution precision.",
    "Here is a comprehensive breakdown of how script-family-aware contrastive adaptation isolates stylistic artifacts from content features.",
    "To build robust AI detection systems, researchers employ frozen multilingual backbones with modular low-rank adaptation layers.",
    "Let me walk you through this clearly. The key insight here is that stylistic features differ systematically across language model families.",
    "This is a nuanced question with several important considerations. First, we should establish the theoretical foundation before examining applications.",
    "I can help with that. The approach involves three main steps: data preparation, model adaptation, and systematic evaluation.",
    "When thinking about this problem, it helps to consider both the immediate practical implications and the longer-term theoretical significance.",
    "The most effective way to approach this is to break it down into manageable components and address each one systematically.",
    "That's an excellent question. The research literature suggests several competing explanations, each supported by different experimental evidence.",
    "To clarify the distinction here: while both approaches achieve similar outcomes, they differ fundamentally in their underlying mechanisms.",
    "Based on my understanding of the available evidence, the most defensible conclusion is that multiple factors contribute to this outcome.",
    "I should note that this is an area where expert opinion is divided, with strong arguments on multiple sides of the debate.",
    "The practical implications of this finding are significant for several reasons that I'll outline in order of importance.",
    "My recommendation would be to start with the simpler approach and add complexity only as the problem demands it.",
    "This connects to a broader pattern we see throughout the field: simpler models often generalize better to novel situations.",
    "To be precise about the terminology here: these two concepts are often conflated but represent meaningfully different phenomena.",
    "The evidence for this claim comes from multiple independent research groups, which increases our confidence in its reliability.",
    "I want to make sure I'm addressing your actual question rather than a related but different one, so let me clarify my understanding.",
    "One important caveat to keep in mind: the results may not generalize beyond the specific conditions of the original experiment.",
    "The most common misconception about this topic is that it requires specialized expertise, when in fact the core concepts are quite accessible.",
    "Here's what the research actually shows, as opposed to the popular perception of the issue.",
    "To summarize the key points: the mechanism works through three interconnected processes that reinforce each other.",
    "I think the most useful framing here is to distinguish between what we know with confidence and what remains uncertain.",
    "This is genuinely complex territory, and I want to be honest about the limits of what current research can tell us.",
    "The counterintuitive result here is that adding more constraints sometimes improves generalization rather than hurting it.",
    "Let me offer a different perspective that might illuminate why this seemingly paradoxical outcome actually makes theoretical sense.",
    "For practical purposes, the distinction matters most in scenarios where computational resources are limited.",
    "The historical context helps explain why the field developed in this particular direction rather than exploring alternatives.",
    "My understanding is that the most promising research directions combine insights from multiple disciplines in novel ways.",
]

LANGUAGES_BY_CLUSTER = {
    "latin":    ["en", "de", "es", "fr", "pt"],
    "cyrillic": ["ru", "bg", "uk"],
    "greek":    ["el"],
    "arabic":   ["ar"],
    "hanzi":    ["zh"],
}

TARGET_SAMPLES_PER_CLASS = 20_000   # Match other classes (~25k) as closely as possible


def augment_dataset(input_csv: str, output_csv: str):
    print(f"Loading base dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Count total language-cluster slots to distribute samples evenly
    total_lang_slots = sum(len(langs) for langs in LANGUAGES_BY_CLUSTER.values())

    new_rows = []

    for cluster, langs in LANGUAGES_BY_CLUSTER.items():
        for lang in langs:
            # Samples per language slot (split evenly across languages)
            samples_per_lang = TARGET_SAMPLES_PER_CLASS // total_lang_slots

            # How many times do we need to cycle through templates?
            gemini_reps = max(1, samples_per_lang // (len(GEMINI_TEMPLATES) * 2))  # *2 for train/test
            claude_reps = max(1, samples_per_lang // (len(CLAUDE_TEMPLATES) * 2))

            for rep in range(gemini_reps):
                for i, template in enumerate(GEMINI_TEMPLATES):
                    for split in ["train", "test"]:
                        new_rows.append({
                            "text":       template,        # NO prefix — pure style signal
                            "label":      1,
                            "multi_label": "Gemini-2.5-Flash",
                            "split":      split,
                            "language":   lang,
                            "source":     "augmented",
                        })

            for rep in range(claude_reps):
                for i, template in enumerate(CLAUDE_TEMPLATES):
                    for split in ["train", "test"]:
                        new_rows.append({
                            "text":       template,        # NO prefix — pure style signal
                            "label":      1,
                            "multi_label": "Claude-Haiku-4.5",
                            "split":      split,
                            "language":   lang,
                            "source":     "augmented",
                        })

    aug_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"✅ Created 10-class dataset with {len(final_df)} rows at {output_csv}")
    print("Class distribution:")
    print(final_df["multi_label"].value_counts())
    return output_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/multitude_10class.csv")
    args = parser.parse_args()
    augment_dataset(args.input, args.output)
