"""
repair_forest.py
================
Entity-driven multi-hop query composition pipeline.

For multi-clause questions:
  1. Split on connectives (order-agnostic — known clause can be on either side)
  2. Resolve the known clause via exact memory or soft match
  3. Extract the entity the answer NEWLY introduces (novelty-based, not the
     first entity found, which often re-states the question's subject)
  4. Substitute the novel entity into the residual clause
  5. Look up the composed question
  6. Recurse while unresolved connectives remain (supports 3+ hop chains)

Design decisions (from testing, Section 2 of design doc):
  - Order-agnostic splitting: the known clause can appear on either side of
    a connective, so both orderings are tried.
  - Novelty-based entity selection: pick the entity the answer *introduces*,
    not the first entity (which often echoes the question subject).
  - Wrong-hop propagation is left unaddressed by design: within the full
    architecture such an edge is rarely exercised and never positively
    reinforced, so it decays naturally rather than needing an explicit fix.
"""

from __future__ import annotations
import re
from typing import Optional

from repair_lm.memory import LongTermMemory
from repair_lm.entity_extractor import EntityExtractor
from repair_lm.soft_match import SoftMatcher


# Connectives that can split a multi-clause question
CONNECTIVES: list[str] = [
    " and ", " who ", " which ", " that ", " whose ", " whom ",
]

# Referring phrases that get substituted in the residual clause
REFERRING_PATTERNS: list[str] = [
    r"\bwho\b", r"\bwhich\b", r"\bthat\b", r"\bwhat\b",
    r"\bit\b",  r"\bthey\b",  r"\bthe answer\b", r"\b_\b",
]

MAX_DEPTH: int = 6   # recursion guard for 3+ hop chains


class QueryComposer:
    """
    Recursive multi-hop query composition.

    Resolution order per clause:
      1. Exact memory lookup (via LongTermMemory)
      2. Soft first-clause matching (via SoftMatcher)

    Composition loop:
      Split → resolve known clause → extract novel entity → substitute →
      recurse on composed question → repeat until resolved or depth exceeded.
    """

    def __init__(self,
                 memory:  LongTermMemory,
                 soft:    SoftMatcher) -> None:
        self._memory    = memory
        self._soft      = soft
        self._extractor = EntityExtractor()

    # ── public API ────────────────────────────────────────────────────────────

    def compose(self, question: str, depth: int = 0) -> Optional[str]:
        """
        Recursively resolve `question`.  Returns the final answer string,
        or None if no resolution path succeeds.
        """
        if depth > MAX_DEPTH:
            return None

        # Base case: direct resolution
        answer = self._resolve(question)
        if answer:
            return answer

        # Recursive case: try splitting on each connective
        for conn in CONNECTIVES:
            idx = question.lower().find(conn)
            if idx == -1:
                continue

            left  = question[:idx].strip()
            right = question[idx + len(conn):].strip()
            if not left or not right:
                continue

            # Try each ordering (known clause on left OR right)
            for known, residual in [(left, right), (right, left)]:
                result = self._compose_step(known, residual, depth)
                if result:
                    return result

        return None

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve(self, clause: str) -> Optional[str]:
        """Try exact memory, then soft match."""
        answer = self._memory.lookup(clause)
        if answer:
            return answer
        result = self._soft.match(clause)
        return result[0] if result else None

    def _compose_step(self,
                      known:    str,
                      residual: str,
                      depth:    int) -> Optional[str]:
        """
        One composition step:
          1. Resolve the known clause.
          2. Extract the novel entity from the answer.
          3. Substitute into the residual clause.
          4. Recurse.
        """
        answer = self._resolve(known)
        if not answer:
            return None

        # Novel entity: NOT already in the known clause (novelty-based selection)
        known_entities = self._extractor.extract(known)
        novel = self._extractor.extract_novel(answer, known_entities)
        if not novel:
            return None

        composed = self._substitute(residual, novel)
        return self.compose(composed, depth + 1)

    def _substitute(self, clause: str, entity: str) -> str:
        """
        Replace the leftmost referring phrase in `clause` with `entity`.
        If no referring phrase is found, prepend the entity.

        Uses leftmost-first substitution (count=1) to avoid over-replacement
        in clauses with multiple pronouns.
        """
        for pattern in REFERRING_PATTERNS:
            new_clause, n = re.subn(
                pattern, entity, clause, count=1, flags=re.IGNORECASE
            )
            if n:
                return new_clause
        # No referring phrase — prepend entity
        return f"{entity} {clause}"
