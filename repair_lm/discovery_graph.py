"""
discovery_graph.py
==================
Discovery-graph fallback: word-overlap similarity search over all memorised
questions.

Used only when memory lookup and composition both fail.  Lower precision than
soft matching (no predicate gate, no entity bonus, no margin requirement) — it
is the last resort before returning no answer.

The "graph" metaphor: nodes are memorised questions; edges connect questions
that share content words.  As the memory grows, the graph grows — more nodes,
more edges, better coverage.  With enough data, even rare phrasings find a
nearby node to fall back on.
"""

from __future__ import annotations
import re
from typing import Optional

from repair_lm.memory import LongTermMemory


STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "which",
    "where", "when", "why", "how", "in", "on", "at", "by", "for", "with",
    "about", "to", "from", "of", "and", "or", "but", "not", "it", "its",
    "did", "do", "does", "have", "has", "had", "be", "been",
}

MIN_OVERLAP: float = 0.05   # discard candidates with zero meaningful overlap


class DiscoveryGraph:
    """
    Brute-force word-overlap similarity search over the memorised question set.

    Time complexity: O(|memory| × |query_tokens|) — acceptable for small-to-medium
    memories; replace with an inverted index if memory exceeds ~100k facts.
    """

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    # ── public API ────────────────────────────────────────────────────────────

    def best_match(self, query: str) -> Optional[tuple[str, float]]:
        """
        Return (answer, score) for the best-overlapping memorised question,
        or None if no candidate reaches MIN_OVERLAP.
        """
        results = self.search(query, top_k=1)
        if not results:
            return None
        _, answer, score = results[0]
        return answer, score

    def search(self, query: str,
               top_k: int = 5) -> list[tuple[str, str, float]]:
        """
        Return up to top_k (question, answer, overlap_score) triples,
        sorted descending by score.
        """
        qw = self._content_words(query)
        if not qw:
            return []

        results: list[tuple[str, str, float]] = []
        for q, a in self._memory.all_items():
            cw = self._content_words(q)
            if not cw:
                continue
            score = len(qw & cw) / len(qw | cw)   # Jaccard
            if score >= MIN_OVERLAP:
                results.append((q, a, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    # ── helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _content_words(text: str) -> set[str]:
        tokens = re.findall(r"\w+", text.lower())
        return {t for t in tokens if t not in STOP_WORDS}
