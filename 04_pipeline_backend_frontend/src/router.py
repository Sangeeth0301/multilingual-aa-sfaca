"""Unicode-block script router (no torch dependency)."""
import re


def detect_script(text: str) -> dict:
    if not text:
        return {"cluster": "latin", "counts": {}, "routing_confidence": 0.0, "mixed": False, "supported": False}

    # Unicode-block based counts, ignoring digits, punctuation, etc.
    cyrillic = len(re.findall(r"[\u0400-\u04FF\u0500-\u052F]", text))
    greek = len(re.findall(r"[\u0370-\u03FF\u1F00-\u1FFF]", text))
    arabic = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))
    hanzi = len(re.findall(r"[\u4E00-\u9FFF]", text))
    kana = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF]", text))
    latin = len(re.findall(r"[\u0041-\u005A\u0061-\u007A\u00C0-\u024F\u1E00-\u1EFF]", text))
    hangul = len(re.findall(r"[\uAC00-\uD7AF\u1100-\u11FF]", text))
    hebrew = len(re.findall(r"[\u0590-\u05FF]", text))
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))

    counts = {
        "cyrillic": cyrillic,
        "greek": greek,
        "arabic": arabic,
        "hanzi": hanzi + kana + hangul,  # Map Japanese/Korean to nearest cluster (Hanzi)
        "latin": latin,
        "unsupported": hebrew + devanagari
    }

    # Filter out empty clusters
    active_counts = {k: v for k, v in counts.items() if v > 0}
    total = sum(active_counts.values())
    
    if total == 0:
        return {"cluster": "latin", "counts": {}, "routing_confidence": 0.0, "mixed": False, "supported": False}

    best = max(counts, key=counts.get)
    supported = (best != "unsupported")
    routing_confidence = counts[best] / total
    mixed = len([c for c, v in active_counts.items() if c != "unsupported"]) > 1
    
    if not supported:
        best = "latin"
        
    return {
        "cluster": best,
        "counts": active_counts,
        "routing_confidence": round(routing_confidence, 4),
        "mixed": mixed,
        "supported": supported
    }
