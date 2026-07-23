"""
fact_corpus.py
==============
Manages the fact corpus used for relation clustering and soft matching.

Each fact is reduced to a (template, value) pair:
  template : the question with its main entity abstracted to the placeholder 'E'
             e.g. "what nationality was Marie Curie" → "what nationality was E"
  value    : the novel entity the answer contributes (not the question's subject)
             e.g. "She was Polish" → "polish"

This representation is what the data-driven relation clustering operates on:
two templates are inferred to express the same relation if they produce
overlapping values across the corpus.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fact:
    """A single Q/A fact with its derived template and value."""
    question: str
    answer:   str
    template: str = field(default="", init=False)
    value:    str = field(default="", init=False)

    def __post_init__(self) -> None:
        # Imported here to avoid circular imports at module level
        from repair_lm.entity_extractor import EntityExtractor
        ex = EntityExtractor()

        # Template: abstract the primary entity in the question
        entity = ex.extract_primary(self.question)
        self.template = (ex.abstract_entity(self.question, entity)
                         if entity else self.question)

        # Value: the novel entity the answer introduces
        q_entities   = ex.extract(self.question)
        novel        = ex.extract_novel(self.answer, q_entities)
        self.value   = (novel or self.answer).lower().strip()

    def __repr__(self) -> str:
        return (f"Fact(template={self.template!r}, "
                f"value={self.value!r})")


class FactCorpus:
    """
    Append-only collection of facts with template/value pair indexing.

    Used by:
      - RelationClustering  (reads template_value_pairs)
      - SoftMatcher         (reads via LongTermMemory)
      - DiscoveryGraph      (reads via LongTermMemory)
    """

    def __init__(self) -> None:
        self._facts: list[Fact] = []

    # ── mutation ──────────────────────────────────────────────────────────────

    def add(self, question: str, answer: str) -> Fact:
        fact = Fact(question=question, answer=answer)
        self._facts.append(fact)
        return fact

    def add_many(self, pairs: list[tuple[str, str]]) -> None:
        for q, a in pairs:
            self.add(q, a)

    # ── access ────────────────────────────────────────────────────────────────

    def all_facts(self) -> list[Fact]:
        return list(self._facts)

    def template_value_pairs(self) -> list[tuple[str, str]]:
        """Return (template, value) pairs for all facts — input to clustering."""
        return [(f.template, f.value) for f in self._facts]

    def templates(self) -> list[str]:
        return [f.template for f in self._facts]

    def values(self) -> list[str]:
        return [f.value for f in self._facts]

    # ── dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"FactCorpus({len(self._facts)} facts)"
