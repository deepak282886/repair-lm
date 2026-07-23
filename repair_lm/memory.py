"""
memory.py
=========
Long-term memory: exact (normalised) question → answer table.

Checked first on every query. Once stored, a fact is never overwritten or
degraded by later additions — the memory is append-only and deterministic.
Normalisation (lowercase, strip punctuation, collapse whitespace) ensures
minor surface variants of the same question hit the same entry.
"""

import re
from typing import Optional


class LongTermMemory:
    """
    Exact normalised question → answer lookup table.

    Design properties
    -----------------
    - Append-only: storing a question that already exists is a no-op.
    - Normalised lookup: "Who wrote Hamlet?" and "who wrote hamlet" both hit
      the same key.
    - O(1) lookup via dict.
    """

    def __init__(self):
        self._store:    dict[str, str] = {}   # normalised_q → answer
        self._raw_keys: dict[str, str] = {}   # normalised_q → original question

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def normalise(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    # ── read / write ──────────────────────────────────────────────────────────

    def store(self, question: str, answer: str) -> bool:
        """
        Store a Q/A fact.  Returns True if stored, False if already present
        (existing entry is never overwritten).
        """
        key = self.normalise(question)
        if key in self._store:
            return False
        self._store[key]    = answer.strip()
        self._raw_keys[key] = question.strip()
        return True

    def lookup(self, question: str) -> Optional[str]:
        """Return answer for exact normalised match, or None."""
        return self._store.get(self.normalise(question))

    # ── bulk access ───────────────────────────────────────────────────────────

    def all_questions(self) -> list[str]:
        """Return all normalised question keys."""
        return list(self._store.keys())

    def all_items(self) -> list[tuple[str, str]]:
        """Return list of (normalised_question, answer) pairs."""
        return list(self._store.items())

    def raw_question(self, normalised: str) -> Optional[str]:
        """Recover the original (un-normalised) question for a key."""
        return self._raw_keys.get(normalised)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, question: str) -> bool:
        return self.normalise(question) in self._store

    def __repr__(self) -> str:
        return f"LongTermMemory({len(self._store)} facts)"
