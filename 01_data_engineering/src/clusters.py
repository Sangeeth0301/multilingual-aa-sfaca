"""
clusters.py
-----------
Defines the mapping from language code -> script/family cluster.

This is the core design choice behind SFA-CA: instead of training one
adapter for every language, we group languages into a small number of
script/family clusters and train ONE adapter per cluster.

The 18 languages below match the base paper's MULTITuDE v3 selection
(La Cava et al., ACL 2026, Table 1 / Figure 1). Low-resource languages
you add later should be appended to LOW_RESOURCE_LANGUAGES with your
own best-guess cluster assignment (see assign_cluster_for_new_language).
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Base paper's 18 languages, grouped by writing script
#    (script clusters are what SFA-CA trains separate adapters for)
# ---------------------------------------------------------------------------

SCRIPT_CLUSTERS = {
    "latin": [
        "nl", "en", "de",              # Germanic
        "pt", "ro", "es",              # Romance
        "hr", "cs", "pl", "sk", "sl",  # Slavic-Latin
        "hu",                          # Uralic
        "ca", "ga", "gd",              # Catalan, Irish, Scottish Gaelic
                                       # (present in the raw MULTITuDE v3
                                       # file's 21 languages, but NOT part
                                       # of the base paper's selected 18 --
                                       # kept here so the code doesn't crash
                                       # on them; excluded via BASE_PAPER_18
                                       # below when you want exact parity)
    ],
    "cyrillic": [
        "bg", "uk", "ru",              # Slavic-Cyrillic
    ],
    "greek": [
        "el",                          # Hellenic
    ],
    "arabic": [
        "ar",                          # Semitic
    ],
    "hanzi": [
        "zh",                          # Sino-Tibetan
    ],
}

# The exact 18 languages used in the base paper (La Cava et al., ACL 2026,
# Table 1) -- use this list to filter the raw 21-language file down to
# match their setup exactly, for direct, apples-to-apples comparison.
BASE_PAPER_18_LANGUAGES = [
    "nl", "en", "de",              # Germanic
    "el",                          # Hellenic
    "ar",                          # Semitic
    "zh",                          # Sino-Tibetan
    "bg", "uk", "ru",              # Slavic-Cyrillic
    "hr", "cs", "pl", "sk", "sl",  # Slavic-Latin
    "pt", "ro", "es",              # Romance
    "hu",                          # Uralic
]

# Reverse lookup: language code -> cluster name
LANG_TO_CLUSTER = {
    lang: cluster
    for cluster, langs in SCRIPT_CLUSTERS.items()
    for lang in langs
}


# ---------------------------------------------------------------------------
# 2. Low-resource language extension
#    Add your new languages here once you've picked them (Phase 4 / Exp 3).
#    Use ISO 639-1 codes where possible.
# ---------------------------------------------------------------------------

LOW_RESOURCE_LANGUAGES = {
    # Example entries -- replace with your actual chosen languages.
    # "ta": "tamil",      # Tamil script -> new cluster "tamil"
    # "sw": "latin",      # Swahili uses Latin script -> reuse "latin" cluster
    # "am": "ethiopic",   # Amharic -> new cluster "ethiopic"
}


@dataclass
class ClusterInfo:
    cluster_name: str
    is_new_cluster: bool  # True if this cluster has no adapter trained on
                           # base-paper languages (i.e. genuinely unseen)


def get_cluster(lang_code: str) -> str:
    """Return the script/family cluster for a known language code."""
    if lang_code in LANG_TO_CLUSTER:
        return LANG_TO_CLUSTER[lang_code]
    if lang_code in LOW_RESOURCE_LANGUAGES:
        return LOW_RESOURCE_LANGUAGES[lang_code]
    raise KeyError(
        f"Unknown language code '{lang_code}'. "
        f"Add it to SCRIPT_CLUSTERS or LOW_RESOURCE_LANGUAGES in clusters.py."
    )


def all_clusters() -> list:
    """List of every distinct cluster name currently defined."""
    clusters = set(SCRIPT_CLUSTERS.keys())
    clusters.update(LOW_RESOURCE_LANGUAGES.values())
    return sorted(clusters)


def nearest_cluster_for_unseen(lang_code: str, script_hint: str) -> str:
    """
    Simple nearest-cluster heuristic for Experiment 2 (zero-shot transfer)
    and Experiment 3 (low-resource languages).

    script_hint: a rough label you assign by inspection, e.g. one of
                 "latin", "cyrillic", "greek", "arabic", "hanzi", or
                 something new (e.g. "dravidian").

    If script_hint matches an existing trained cluster, reuse its adapter.
    Otherwise, fall back to the "latin" cluster as a default (or you can
    change this fallback after running your ablation study).
    """
    if script_hint in all_clusters():
        return script_hint
    return "latin"  # fallback default -- revisit after Exp 4 ablation


if __name__ == "__main__":
    # Quick sanity check when you run: python 01_data_engineering/src/clusters.py
    print("Defined clusters:", all_clusters())
    for lang, cluster in LANG_TO_CLUSTER.items():
        print(f"  {lang} -> {cluster}")
