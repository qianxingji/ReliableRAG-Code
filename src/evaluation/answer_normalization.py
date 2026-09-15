from __future__ import annotations

import re
import string
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """HotpotQA/SQuAD-style normalization used for answer equality.

    Lowercase, remove ASCII punctuation, remove English articles, then collapse
    whitespace. This mirrors the normalization described in the accepted manuscript.
    """
    if text is None:
        text = ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class PairEligibility:
    eligible: bool
    reason: str | None
    normalized_a0: str
    normalized_a1: str


def assess_pair_eligibility(a0: str, a1: str) -> PairEligibility:
    """Apply the shared fail-closed post-repair eligibility rule."""
    raw0 = "" if a0 is None else str(a0)
    raw1 = "" if a1 is None else str(a1)
    if not raw0.strip():
        return PairEligibility(False, "a0_empty", "", normalize_answer(raw1))
    if not raw1.strip():
        return PairEligibility(False, "a1_empty", normalize_answer(raw0), "")
    n0 = normalize_answer(raw0)
    n1 = normalize_answer(raw1)
    if n0 == n1:
        return PairEligibility(False, "normalized_answers_equal", n0, n1)
    return PairEligibility(True, None, n0, n1)
