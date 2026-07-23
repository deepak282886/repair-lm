"""
predicate_gate.py
=================
Determines whether two clauses are asking about the "same kind of thing".
Used by soft matching to suppress false positives where query and candidate
share surface words or an entity but are recognisably different relations.

e.g. "who discovered the Ninth Symphony" must NOT match
     "who composed the Ninth Symphony" — different predicate groups.

Two implementations are available (selected automatically):

  1. Hand-written keyword → group dictionary (production default)
     - Works immediately, no corpus required
     - Silent failure on predicates outside the fixed vocabulary
     - Grows manually as new relation types appear

  2. Data-driven RelationClustering (used when clustering is fitted)
     - Infers groups from corpus co-occurrence evidence
     - Falls back to hand-list when clustering has no evidence for a template
     - Conservative: novel structures get no protection rather than wrong protection

From Section 4/6 of the design doc:
  The hand-list raises false-positive refusal from 0/3 → 3/3.
  Clustering v2 (after entity-extractor fix) raises it to 2/3 at 47 facts.
  Recommended: keep hand-list as production gate; re-run clustering as corpus
  grows toward 50-100 facts.
"""

from __future__ import annotations
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from repair_lm.relation_clustering import RelationClustering


# ── hand-written predicate groups ─────────────────────────────────────────────

PREDICATE_GROUPS: dict[str, str] = {
    # authorship
    "wrote":      "authorship",
    "written":    "authorship",
    "author":     "authorship",
    "authored":   "authorship",
    "write":      "authorship",
    # composition (musical)
    "composed":   "composition",
    "compose":    "composition",
    "composer":   "composition",
    "composition":"composition",
    # discovery / finding
    "discovered": "discovery",
    "discover":   "discovery",
    "discovery":  "discovery",
    "found":      "discovery",
    "identified": "discovery",
    # invention
    "invented":   "invention",
    "invent":     "invention",
    "invention":  "invention",
    "inventor":   "invention",
    "created":    "invention",
    # nationality / origin
    "nationality":"nationality",
    "national":   "nationality",
    "country":    "nationality",
    # birthplace
    "born":       "birthplace",
    "birthplace": "birthplace",
    "birth":      "birthplace",
    "hometown":   "birthplace",
    # location / region
    "located":    "location",
    "location":   "location",
    "situated":   "location",
    "city":       "location",
    "capital":    "location",
    "where":      "location",
    "state":      "region",
    "region":     "region",
    "province":   "region",
    # direction (film)
    "directed":   "direction",
    "director":   "direction",
    "directs":    "direction",
    # visual art
    "painted":    "painting",
    "painter":    "painting",
    "drew":       "painting",
    "illustrated":"painting",
    # identity / role
    "played":     "role",
    "starred":    "role",
    "acted":      "role",
    "portrayed":  "role",
}

# Score multiplier applied when predicates are from different known groups
MISMATCH_PENALTY: float = 0.4


class PredicateGate:
    """
    Compute a score multiplier for a (query, candidate) pair.

    Returns 1.0 (no penalty) when:
      - Predicates are from the same group
      - Either predicate is unknown (conservative — no false suppression)

    Returns MISMATCH_PENALTY when:
      - Predicates are from different known groups
    """

    def __init__(self,
                 clustering: Optional["RelationClustering"] = None) -> None:
        """
        Parameters
        ----------
        clustering : RelationClustering | None
            If provided and fitted, used as primary gate.
            Falls back to hand-written groups when clustering has no evidence.
        """
        self._clustering = clustering

    # ── public API ────────────────────────────────────────────────────────────

    def mismatch_penalty(self, query: str, candidate: str) -> float:
        """
        Return a score multiplier for the candidate given the query.
        1.0 = no penalty | MISMATCH_PENALTY = penalise.
        """
        # Try data-driven clustering first if available and fitted
        if self._clustering is not None and self._clustering.is_fitted():
            qt = self._to_template(query)
            ct = self._to_template(candidate)
            qt_known = qt in self._clustering._template_to_root
            ct_known = ct in self._clustering._template_to_root
            if qt_known and ct_known:
                return (1.0 if self._clustering.same_relation(qt, ct)
                        else MISMATCH_PENALTY)
            # Fall through if clustering lacks evidence for either template

        # Hand-written fallback
        qg = self._predicate_group(query)
        cg = self._predicate_group(candidate)
        if qg is not None and cg is not None and qg != cg:
            return MISMATCH_PENALTY
        return 1.0

    # ── private helpers ───────────────────────────────────────────────────────

    def _predicate_group(self, text: str) -> Optional[str]:
        """Return the predicate group for text using the hand-written dict."""
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            if tok in PREDICATE_GROUPS:
                return PREDICATE_GROUPS[tok]
        return None

    @staticmethod
    def _to_template(text: str) -> str:
        """
        Rough template extraction for clustering lookup:
        lowercase + abstract capitalised spans to 'E'.
        """
        t = re.sub(
            r"[A-Z][a-z]+(?:\s+(?:da|de|van|von|du)\s+[A-Z][a-z]+)*"
            r"(?:\s+[A-Z][a-z]+)*",
            "E", text
        )
        return t.lower().strip()
