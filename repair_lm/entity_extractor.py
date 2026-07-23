"""
entity_extractor.py
===================
Identifies subject/object spans in questions and answers.

Priority
--------
1. Capitalised proper-noun spans, including multi-word names with lowercase
   particles ('da', 'van', 'von', 'de', 'du', …).
   e.g. "Leonardo da Vinci", "Ludwig van Beethoven", "Charles de Gaulle"

2. Fallback to the last content word when no capitalised entity is present.
   Needed for facts whose object is a lowercase common noun:
   e.g. "who discovered penicillin" → "penicillin"
        "what is the capital of france" → "france"

Novelty-based entity selection
-------------------------------
extract_novel() returns the entity in an answer that is NOT already mentioned
in the question — the new information the answer contributes.  This is used
by the composition pipeline to pick which entity to substitute into the
residual clause of a multi-hop question (not just the first entity found,
which often re-states the question's subject).
"""

import re
from typing import Optional

# Lowercase particles that can appear within a proper-noun span
PARTICLES: set[str] = {
    "da", "de", "del", "della", "des", "di", "du", "el", "la", "le",
    "los", "van", "van der", "van den", "von", "af", "av", "of", "the",
    "al", "bin", "bint", "ap", "ab",
}

# Stop words excluded from the content-word fallback
STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "what", "who", "which",
    "where", "when", "why", "how", "in", "on", "at", "by", "for", "with",
    "about", "to", "from", "of", "and", "or", "but", "not", "no", "it",
    "its", "this", "that", "these", "those", "my", "your", "his", "her",
    "their", "our", "did", "was", "were", "been",
}


class EntityExtractor:
    """
    Extract entity spans from natural-language questions and answers.
    """

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[str]:
        """
        Return all entity spans found in text, in order of appearance.
        Falls back to [last_content_word] if no capitalised spans found.
        """
        entities = self._extract_capitalised(text)
        if not entities:
            fb = self._last_content_word(text)
            if fb:
                entities = [fb]
        return entities

    def extract_primary(self, text: str) -> Optional[str]:
        """
        Return the single most salient entity (first capitalised span,
        or last content word if none).
        """
        entities = self.extract(text)
        return entities[0] if entities else None

    def extract_novel(self, answer: str,
                      question_entities: list[str]) -> Optional[str]:
        """
        Return the first entity in `answer` that does NOT appear in
        `question_entities`.  This is the new information the answer
        introduces — used for novelty-based entity selection in multi-hop
        composition.

        If every entity in the answer re-states a question entity, fall back
        to the last content word of the answer.
        """
        q_set = {e.lower() for e in question_entities}
        for entity in self.extract(answer):
            if entity.lower() not in q_set:
                return entity
        return self._last_content_word(answer)

    def abstract_entity(self, text: str, entity: str,
                        placeholder: str = "E") -> str:
        """Replace `entity` in `text` with `placeholder` (case-insensitive)."""
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        return pattern.sub(placeholder, text)

    # ── private helpers ───────────────────────────────────────────────────────

    def _clean(self, token: str) -> str:
        """Strip trailing/leading punctuation from a token."""
        return re.sub(r"^[^\w]+|[^\w]+$", "", token)

    def _extract_capitalised(self, text: str) -> list[str]:
        """
        Extract spans of capitalised tokens, merging across lowercase particles.

        Algorithm:
        - Walk tokens left to right.
        - When a capitalised token is found, start a span.
        - Extend the span if the next token is also capitalised, or if it is a
          particle immediately followed by a capitalised token (lookahead).
        - Emit the span when the extension rules no longer apply.
        """
        tokens  = text.split()
        entities: list[str] = []
        i = 0
        while i < len(tokens):
            tok = self._clean(tokens[i])
            if tok and tok[0].isupper():
                span = [tok]
                j = i + 1
                while j < len(tokens):
                    next_tok = self._clean(tokens[j])
                    if next_tok and next_tok[0].isupper():
                        span.append(next_tok)
                        j += 1
                    elif next_tok.lower() in PARTICLES and j + 1 < len(tokens):
                        # Include particle only if followed by capitalised token
                        after = self._clean(tokens[j + 1])
                        if after and after[0].isupper():
                            span.append(next_tok)
                            j += 1          # particle consumed; loop will grab 'after'
                        else:
                            break
                    else:
                        break
                entities.append(" ".join(span))
                i = j
            else:
                i += 1
        return entities

    def _last_content_word(self, text: str) -> Optional[str]:
        """Return the last non-stop word in the text (lowercase)."""
        tokens = [re.sub(r"[^\w]", "", t).lower() for t in text.split()]
        for tok in reversed(tokens):
            if tok and tok not in STOP_WORDS:
                return tok
        return None
