"""
sanitizer.py
------------
Math & Notation-Aware Text Sanitizer and Confidence Calibrator for SFA-CA.

Real-world AI detectors frequently produce false-positive AI flags on human-written
academic and technical documents due to LaTeX equations, code blocks, and mathematical
notations (e.g., $E = mc^2$, \\begin{equation}...\\end{equation}).

This module:
1. Detects and extracts LaTeX math blocks, markdown code snippets, and chemical/math notations.
2. Sanitizes the text before passing it to XLM-RoBERTa for attribution.
3. Computes a Notation Density Index (NDI).
4. Calibrates raw Softmax probabilities for math-heavy text to eliminate false positives.
"""

import re
from typing import Dict, Tuple, Any, List

LATEX_MATH_PATTERNS = [
    r"\$\$[\s\S]*?\$\$",                  # Display math $$ ... $$
    r"\$[^$\n]+\$",                       # Inline math $ ... $
    r"\\\[[\s\S]*?\\\]",                  # Display math \[ ... \]
    r"\\\([\s\S]*?\\\)",                  # Inline math \( ... \)
    r"\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}",  # Equations
    r"\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}",        # Align blocks
    r"\\begin\{matrix\*?\}[\s\S]*?\\end\{matrix\*?\}",      # Matrices
    r"\\begin\{gather\*?\}[\s\S]*?\\end\{gather\*?\}",      # Gather blocks
]

CODE_BLOCK_PATTERNS = [
    r"```[\s\S]*?```",                    # Fenced code blocks
    r"`[^`\n]+`",                          # Inline code
]


class TextSanitizer:
    """
    Extracts math/code notation and computes Notation Density Index (NDI).
    """

    def __init__(self):
        self.math_regex = re.compile("|".join(LATEX_MATH_PATTERNS))
        self.code_regex = re.compile("|".join(CODE_BLOCK_PATTERNS))

    def sanitize(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Sanitizes input text by removing raw LaTeX math and code blocks.
        
        Returns:
            sanitized_text: String with math/code blocks replaced by space placeholders.
            meta: Dictionary with extracted math/code snippets and Notation Density Index (NDI).
        """
        if not raw_text or not raw_text.strip():
            return "", {"ndi": 0.0, "extracted_math": [], "extracted_code": [], "math_char_count": 0}

        original_len = len(raw_text)

        # 1. Extract math snippets
        math_matches = self.math_regex.findall(raw_text)
        text_no_math = self.math_regex.sub(" ", raw_text)

        # 2. Extract code snippets
        code_matches = self.code_regex.findall(text_no_math)
        sanitized_text = self.code_regex.sub(" ", text_no_math)

        # Clean up whitespace
        sanitized_text = re.sub(r"\s+", " ", sanitized_text).strip()

        # Compute Notation Density Index (NDI)
        math_chars = sum(len(m) for m in math_matches) + sum(len(c) for c in code_matches)
        ndi = min(1.0, math_chars / max(1, original_len))

        meta = {
            "ndi": round(ndi, 4),
            "extracted_math": math_matches,
            "extracted_code": code_matches,
            "math_char_count": math_chars,
            "has_math_notation": len(math_matches) > 0 or len(code_matches) > 0,
        }

        # Fallback if text was purely math equations: keep original text
        if len(sanitized_text) < 10 and original_len >= 10:
            sanitized_text = raw_text

        return sanitized_text, meta


def calibrate_probabilities(
    probs: Dict[str, float],
    ndi: float,
    human_class: str = "human",
    calibration_factor: float = 0.35
) -> Dict[str, float]:
    """
    Calibrates Softmax attribution probabilities when Notation Density Index (NDI) is high.
    
    If text contains heavy LaTeX/math notation (high NDI), raw AI attribution scores
    are adjusted to prevent false-positive AI flags on human-written academic papers.
    """
    if ndi <= 0.05 or human_class not in probs:
        return probs

    calibrated = probs.copy()
    
    # Scale boost for human class proportional to math density
    boost = ndi * calibration_factor
    
    # Softly shift probability mass toward human class for math-heavy content
    current_human = calibrated[human_class]
    new_human = min(0.98, current_human + (1.0 - current_human) * boost)
    
    # Renormalize remaining AI generator classes
    scale_factor = (1.0 - new_human) / max(1e-6, 1.0 - current_human)
    for k in calibrated:
        if k == human_class:
            calibrated[k] = round(new_human, 4)
        else:
            calibrated[k] = round(calibrated[k] * scale_factor, 4)

    return calibrated


if __name__ == "__main__":
    # Unit Test
    test_text = r"""
    The energy-momentum relation in special relativity is given by:
    $$E^2 = (pc)^2 + (m_0 c^2)^2$$
    where $p$ represents momentum and $m_0$ is the rest mass.
    """
    sanitizer = TextSanitizer()
    clean_txt, metadata = sanitizer.sanitize(test_text)
    print("Original Text:", test_text)
    print("Sanitized Text:", clean_txt)
    print("Metadata:", metadata)


_MATH_RE = re.compile("|".join(LATEX_MATH_PATTERNS))
_CODE_RE = re.compile("|".join(CODE_BLOCK_PATTERNS))


def find_notation_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return non-overlapping (start, end, type) spans of math/code in `text`.

    Offsets refer to the original string so the UI can highlight in place.
    Code spans win over math spans when they overlap (a `$` inside a code block
    is not LaTeX).
    """
    spans = [(m.start(), m.end(), "code") for m in _CODE_RE.finditer(text)]
    for m in _MATH_RE.finditer(text):
        if not any(m.start() < e and m.end() > s for s, e, _ in spans):
            spans.append((m.start(), m.end(), "math"))
    return sorted(spans)
