"""
soft_match.py
=============
Soft first-clause matching.

When exact memory lookup misses, candidates are scored by:
  content-word overlap  (Jaccard similarity over non-stop-words)
  + entity-presence bonus  (query's primary entity appears in candidate)
  × predicate-mismatch penalty  (from PredicateGate)

Gated by:
  absolute score threshold   (candidate must exceed SCORE_THRESHOLD)
  margin over second-best    (best must beat second by at least MARGIN)

This combination recovers legitimate paraphrases while suppressing false
positives like "who discovered the Ninth Symphony" matching
"who composed the Ninth Symphony" (shared entity + shared words, wrong predicate).

From Section 4 of the design doc:
  - Without the predicate gate: 2/3 false positives passed through
  - With the hand-written gate: 3/3 false positives refused, 3/3 paraphrases
    recovered
  - Thresholds below are the values that achieved this result
"""

from __future__ import annotations
import re
from typing import Optional

from repair_lm.memory import LongTermMemory
from repair_lm.entity_extractor import EntityExtractor
from repair_lm.predicate_gate import PredicateGate


# Tunable thresholds (calibrated on the test set in Section 4)
SCORE_THRESHOLD: float = 0.30   # absolute minimum to accept a candidate
MARGIN:          float = 0.15   # best must exceed second-best by at least this
ENTITY_BONUS:    float = 0.30   # added when query entity appears in candidate

STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "which",
    "where", "when", "why", "how", "in", "on", "at", "by", "for", "with",
    "about", "to", "from", "of", "and", "or", "but", "not", "it", "its",
    "did", "do", "does", "have", "has", "had", "be", "been",
}


class SoftMatcher:
    """
    Score-based approximate question matching with predicate gating.
    """

    def __init__(self,
                 memory: LongTermMemory,
                 gate:   Optional[PredicateGate] = None) -> None:
        self._memory    = memory
        self._extractor = EntityExtractor()
        self._gate      = gate or PredicateGate()

    # ── public API ────────────────────────────────────────────────────────────

    def match(self, query: str) -> Optional[tuple[str, float]]:
        """
        Return (answer, score) for the best candidate that passes all gates,
        or None if no candidate qualifies.

        Scoring pipeline per candidate:
          1. Jaccard similarity over content words
          2. + entity-presence bonus
          3. × predicate-mismatch penalty
          4. Gate: absolute threshold + margin over second-best
        """
        candidates = self._memory.all_items()
        if not candidates:
            return None

        scored = sorted(
            [(q, a, self._score(query, q)) for q, a in candidates],
            key=lambda x: x[2],
            reverse=True,
        )

        best_q, best_a, best_score = scored[0]

        if best_score < SCORE_THRESHOLD:
            return None

        if len(scored) > 1 and (best_score - scored[1][2]) < MARGIN:
            return None

        return best_a, best_score

    def ranked_candidates(self, query: str,
                          top_k: int = 5) -> list[tuple[str, str, float]]:
        """
        Return top-k (normalised_question, answer, score) triples.
        Useful for debugging and evaluation.
        """
        candidates = self._memory.all_items()
        scored = sorted(
            [(q, a, self._score(query, q)) for q, a in candidates],
            key=lambda x: x[2],
            reverse=True,
        )
        return scored[:top_k]

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score(self, query: str, candidate: str) -> float:
        qw = self._content_words(query)
        cw = self._content_words(candidate)
        if not qw or not cw:
            return 0.0

        # Jaccard similarity
        overlap = len(qw & cw) / len(qw | cw)

        # Entity presence bonus
        q_entity = self._extractor.extract_primary(query)
        entity_bonus = (ENTITY_BONUS
                        if q_entity and q_entity.lower() in candidate.lower()
                        else 0.0)

        raw = overlap + entity_bonus

        # Predicate mismatch penalty
        penalty = self._gate.mismatch_penalty(query, candidate)

        return raw * penalty

    @staticmethod
    def _content_words(text: str) -> set[str]:
        tokens = re.findall(r"\w+", text.lower())
        return {t for t in tokens if t not in STOP_WORDS}
