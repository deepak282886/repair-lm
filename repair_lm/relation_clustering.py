"""
relation_clustering.py
======================
Data-driven DIRT-style relation clustering.

Two question templates are inferred to express the same relation if they
resolve to overlapping answer values across the corpus — no synonym
dictionary required.

Examples from testing (Section 5 of design doc):
  - "what nationality was E" + "what country was E from"
      → merged because both produced "english"/"german" for different people
  - "who wrote E" + "who is author of E"
      → stayed separate — no fact happened to give the same author under
        both phrasings (data sparsity at small scale, not a method flaw)
  - "who discovered E"
      → unified across penicillin/radium/gravity/oxygen once lowercase
        common-noun objects were abstracted (entity extractor fix)

Known behaviour (from Section 6):
  - Conservative failure mode: stays silent rather than guessing when
    evidence is absent.  Preferable to the hand-list's silent failure on
    any predicate outside its fixed vocabulary.
  - Requires ~3+ facts per template for reliable clustering signal.
  - Novel sentence structures (never seen in corpus) get zero protection —
    this is an architectural gap, not a bug.

Recommended use: re-run clustering after every significant corpus expansion.
The original document's next step of scaling to 50-100 facts is a natural
checkpoint.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Optional

from repair_lm.fact_corpus import FactCorpus


# ── union-find ────────────────────────────────────────────────────────────────

class UnionFind:
    """
    Path-compressed union-find for clustering string keys.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])   # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def clusters(self, items: list[str]) -> dict[str, list[str]]:
        """Return {root: [members]} for all items."""
        result: dict[str, list[str]] = defaultdict(list)
        for item in items:
            result[self.find(item)].append(item)
        return dict(result)


# ── clustering ────────────────────────────────────────────────────────────────

class RelationClustering:
    """
    Infer relation clusters from (template, value) co-occurrence evidence.

    fit(corpus) must be called before any lookup methods.
    After calling fit(), same_relation() and template_cluster() are available.
    """

    def __init__(self) -> None:
        self._clusters:        Optional[dict[str, list[str]]] = None
        self._template_to_root: dict[str, str]                = {}

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, corpus: FactCorpus) -> None:
        """
        Build clusters from the corpus.

        Algorithm:
          1. Build value → [templates] index from (template, value) pairs.
          2. Union-find: merge all templates that share at least one output value.
          3. Store cluster assignments.
        """
        pairs = corpus.template_value_pairs()

        value_to_templates: dict[str, list[str]] = defaultdict(list)
        for template, value in pairs:
            if value:
                value_to_templates[value].append(template)

        uf = UnionFind()
        for templates in value_to_templates.values():
            for i in range(1, len(templates)):
                uf.union(templates[0], templates[i])

        all_templates          = list({t for t, _ in pairs})
        self._clusters         = uf.clusters(all_templates)
        self._template_to_root = {t: uf.find(t) for t in all_templates}

    # ── lookup ────────────────────────────────────────────────────────────────

    def same_relation(self, template_a: str, template_b: str) -> bool:
        """
        Return True if both templates are in the same cluster.
        Returns False if either template has no corpus evidence (novel structure).
        """
        ra = self._template_to_root.get(template_a)
        rb = self._template_to_root.get(template_b)
        if ra is None or rb is None:
            return False    # novel template → no verdict (architectural gap)
        return ra == rb

    def template_cluster(self, template: str) -> list[str]:
        """Return all templates in the same cluster as `template`."""
        root = self._template_to_root.get(template)
        if root is None or self._clusters is None:
            return []
        return self._clusters.get(root, [])

    def is_fitted(self) -> bool:
        return self._clusters is not None

    # ── inspection ────────────────────────────────────────────────────────────

    def get_clusters(self) -> dict[str, list[str]]:
        return self._clusters or {}

    def print_clusters(self) -> None:
        """Print clusters that contain more than one template."""
        if not self._clusters:
            print("  (not fitted)")
            return
        multi = {r: m for r, m in self._clusters.items() if len(m) > 1}
        if not multi:
            print("  No multi-member clusters found (corpus may be too small).")
            return
        for root, members in multi.items():
            print(f"  Cluster [{root[:50]}]:")
            for m in members:
                print(f"    - {m}")

    def __repr__(self) -> str:
        n = len(self._clusters) if self._clusters else 0
        return f"RelationClustering({n} clusters)"
