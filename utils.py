"""
Shared utilities for the ContextAwareEnv package.
=================================================

Centralises score clamping, fuzzy matching, and keyword scoring
to eliminate code duplication between ``environment.py`` and
``inference.py``.

These functions are intentionally small and pure so they remain
easy to test and reason about.
"""

from __future__ import annotations

import difflib

__all__ = [
    "SCORE_EPSILON",
    "clamp_score",
    "fuzzy_match_score",
    "weighted_keyword_score",
]


# ── Constants ─────────────────────────────────────────────────────────────────

SCORE_EPSILON: float = 0.01
"""Small offset to keep scores strictly inside (0, 1) as required by the
Meta × Scaler evaluation platform."""


# ── Score clamping ────────────────────────────────────────────────────────────

def clamp_score(raw: float) -> float:
    """Clamp *raw* into the open interval (0, 1).

    The grading pipeline rejects scores that are exactly 0.0 or 1.0,
    so we nudge boundary values inward by :data:`SCORE_EPSILON`.

    Examples
    --------
    >>> clamp_score(0.0)
    0.01
    >>> clamp_score(1.0)
    0.99
    >>> clamp_score(0.75)
    0.75
    """
    if raw <= 0.0:
        return SCORE_EPSILON
    if raw >= 1.0:
        return 1.0 - SCORE_EPSILON
    return raw


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def fuzzy_match_score(predicted: str, reference: str) -> float:
    """Calculate a continuous string similarity score between 0.0 and 1.0.

    Uses :func:`difflib.SequenceMatcher` for character-level similarity.
    Both strings are lowercased before comparison.

    Parameters
    ----------
    predicted : str
        The text produced by the agent.
    reference : str
        The ideal / reference text to compare against.

    Returns
    -------
    float
        Similarity ratio in [0.0, 1.0].
    """
    if not predicted:
        return 0.0
    return difflib.SequenceMatcher(
        None, predicted.lower(), reference.lower()
    ).ratio()


# ── Weighted keyword scoring ─────────────────────────────────────────────────

def weighted_keyword_score(text: str, keywords: dict[str, float]) -> float:
    """Score *text* based on weighted keyword presence (case-insensitive).

    Each keyword found in *text* adds its corresponding weight to the
    total.  Keywords not found contribute zero.  This complements
    :func:`fuzzy_match_score` by rewarding mentions of specific terms
    without penalising length differences.

    Parameters
    ----------
    text : str
        The text to evaluate.
    keywords : dict[str, float]
        Mapping of keyword → weight.

    Returns
    -------
    float
        Sum of weights for all keywords found in *text*.

    Examples
    --------
    >>> weighted_keyword_score("npm build error", {"npm": 0.1, "error": 0.1, "webpack": 0.1})
    0.2
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    return sum(
        weight for kw, weight in keywords.items() if kw.lower() in text_lower
    )
