"""
model.py
========
Top-level Re-Pair Language Model.

Combines all components into a unified model with a clean API for
pretraining, finetuning, inference, and persistence.

Query resolution pipeline (checked in order of decreasing confidence):
  1. Exact memory lookup          — deterministic, O(1)
  2. Multi-hop composition        — entity-driven chain traversal
  3. Soft first-clause matching   — gated by predicate/relation check
  4. Discovery-graph fallback     — word-overlap, last resort
  5. Re-Pair grammar generation   — open-ended generation for novel inputs

Training:
  Pretraining : joint Re-Pair over large text corpus → grammar rules
  Finetuning  : QA/instruction pairs → memory + relation clusters + rewards
"""

from __future__ import annotations
import pickle
from typing import Optional

from repair_lm.repair_grammar    import RePairGrammar
from repair_lm.memory            import LongTermMemory
from repair_lm.entity_extractor  import EntityExtractor
from repair_lm.fact_corpus       import FactCorpus
from repair_lm.relation_clustering import RelationClustering
from repair_lm.predicate_gate    import PredicateGate
from repair_lm.soft_match        import SoftMatcher
from repair_lm.repair_forest     import QueryComposer
from repair_lm.discovery_graph   import DiscoveryGraph


# Minimum corpus size before clustering is worth running (from Section 6)
MIN_FACTS_FOR_CLUSTERING: int = 50


class RePairLM:
    """
    Re-Pair Language Model.

    All components are wired together here.  The predicate gate defaults to
    the hand-written dictionary and switches to data-driven clustering once
    refit_clustering() is called with enough facts.
    """

    def __init__(self) -> None:
        # Core components
        self.memory      = LongTermMemory()
        self.corpus      = FactCorpus()
        self.grammar     = RePairGrammar()

        # Relation understanding
        self.clustering  = RelationClustering()
        self.gate        = PredicateGate()               # hand-list default

        # Resolution pipeline
        self.soft        = SoftMatcher(self.memory, self.gate)
        self.composer    = QueryComposer(self.memory, self.soft)
        self.discovery   = DiscoveryGraph(self.memory)

        self._extractor  = EntityExtractor()
        self._finetuned  = False

    # ── resolution pipeline ───────────────────────────────────────────────────

    def answer(self, question: str) -> dict:
        """
        Resolve a question through the full pipeline.

        Returns
        -------
        dict with keys:
          answer : str   — the resolved or generated answer
          source : str   — which component resolved it
          score  : float — confidence (1.0 = exact memory, 0.0 = grammar)
        """
        # 1. Exact memory
        ans = self.memory.lookup(question)
        if ans:
            return {"answer": ans, "source": "memory", "score": 1.0}

        # 2. Multi-hop composition
        ans = self.composer.compose(question)
        if ans:
            return {"answer": ans, "source": "composition", "score": 0.9}

        # 3. Soft matching (with predicate gate)
        result = self.soft.match(question)
        if result:
            return {"answer": result[0], "source": "soft_match",
                    "score": result[1]}

        # 4. Discovery graph
        result = self.discovery.best_match(question)
        if result:
            return {"answer": result[0], "source": "discovery",
                    "score": result[1]}

        # 5. Grammar generation (open-ended fallback)
        generated = self.grammar.generate(prompt=question, max_symbols=30)
        return {"answer": generated, "source": "grammar", "score": 0.0}

    def generate(self, prompt: Optional[str] = None,
                 max_symbols: int = 40) -> str:
        """Open-ended text generation via the Re-Pair grammar."""
        return self.grammar.generate(prompt=prompt, max_symbols=max_symbols)

    # ── learning ──────────────────────────────────────────────────────────────

    def learn_fact(self, question: str, answer: str) -> None:
        """
        Store a single Q/A fact.
        Adds to both long-term memory (for exact lookup) and the fact corpus
        (for relation clustering).
        """
        self.memory.store(question, answer)
        self.corpus.add(question, answer)

    def learn_many(self, pairs: list[tuple[str, str]]) -> None:
        """Store multiple Q/A pairs."""
        for q, a in pairs:
            self.learn_fact(q, a)

    def refit_clustering(self, force: bool = False) -> bool:
        """
        Re-run data-driven relation clustering over the current corpus.

        Switches the predicate gate to use clustering as primary signal
        (falling back to the hand-list when clustering has no evidence).

        Recommended: call after every significant corpus expansion.
        Returns True if clustering was run, False if skipped.

        From Section 6: 50–100 facts is the natural checkpoint; below that,
        data sparsity means clustering underperforms the hand-list.
        """
        if not force and len(self.corpus) < MIN_FACTS_FOR_CLUSTERING:
            print(f"Skipping clustering: {len(self.corpus)} facts "
                  f"(need {MIN_FACTS_FOR_CLUSTERING}+, use force=True to override).")
            return False

        self.clustering.fit(self.corpus)
        # Rebuild pipeline with clustering-aware gate
        self.gate     = PredicateGate(clustering=self.clustering)
        self.soft     = SoftMatcher(self.memory, self.gate)
        self.composer = QueryComposer(self.memory, self.soft)
        self._finetuned = True
        print(f"Clustering updated: {len(self.clustering.get_clusters())} clusters "
              f"over {len(self.corpus)} facts.")
        return True

    # ── reward signal ─────────────────────────────────────────────────────────

    def evaluate(self, pairs: list[tuple[str, str]]) -> dict:
        """
        Run evaluation pairs through the pipeline.
        Returns resolution source counts and exact-match accuracy.
        """
        counts: dict[str, int] = {
            "memory": 0, "composition": 0, "soft_match": 0,
            "discovery": 0, "grammar": 0, "correct": 0,
        }
        for q, gold in pairs:
            out = self.answer(q)
            counts[out["source"]] += 1
            if out["answer"].lower().strip() == gold.lower().strip():
                counts["correct"] += 1
        counts["total"] = len(pairs)
        return counts

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the full model (grammar + memory + clusters) to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}  "
              f"({len(self.memory)} facts, {len(self.grammar.rules)} rules)")

    @staticmethod
    def load(path: str) -> "RePairLM":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded ← {path}  "
              f"({len(model.memory)} facts, {len(model.grammar.rules)} rules)")
        return model

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"RePairLM("
                f"facts={len(self.memory)}, "
                f"rules={len(self.grammar.rules)}, "
                f"finetuned={self._finetuned})")
