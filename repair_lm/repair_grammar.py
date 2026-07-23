"""
repair_grammar.py
=================
Joint Re-Pair compression over a text corpus.

Re-Pair (Larsson & Moffat, 1999) repeatedly replaces the most frequent
adjacent symbol pair with a new non-terminal until no pair exceeds min_freq.
Running it *jointly* over the entire corpus means recurring phrasings across
different sentences (e.g. "what nationality was", "who wrote") become shared
non-terminals — genuine reusable hierarchy rather than per-sentence noise.

Rule frequencies are preserved as unnormalised probabilities, turning the
grammar into a generative model: during generation, rule selection is
weighted by how often each rule was created.
"""

import random
import pickle
from collections import defaultdict, Counter
from typing import Optional


class RePairGrammar:
    """
    Joint Re-Pair probabilistic grammar.

    Terminals    : original vocabulary tokens  (int IDs < NT_START)
    Non-terminals: merged symbols              (int IDs >= NT_START)
    Rules        : non-terminal → (left_symbol, right_symbol)
    rule_freq    : creation frequency → unnormalised generation probability
    """

    NT_START = 10_000   # non-terminals live above this threshold

    def __init__(self):
        self.rules:     dict[int, tuple[int, int]] = {}
        self.rule_freq: Counter                    = Counter()
        self.next_nt:   int                        = self.NT_START
        self.tok2id:    dict[str, int]             = {}
        self.id2tok:    dict[int, str]             = {}
        self._compressed:    list[list[int]]       = []
        self._prefix_index:  Optional[dict[int, list[int]]] = None

    # ── vocabulary ────────────────────────────────────────────────────────────

    def _get_id(self, token: str) -> int:
        if token not in self.tok2id:
            idx = len(self.tok2id)
            self.tok2id[token] = idx
            self.id2tok[idx]   = token
        return self.tok2id[token]

    def _new_nt(self) -> int:
        sym = self.next_nt
        self.next_nt += 1
        return sym

    def sym_label(self, sym: int) -> str:
        if sym in self.id2tok:
            return self.id2tok[sym]
        return f"NT{sym}"

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, sentences: list[str],
              max_rules: int = 500,
              min_freq:  int = 2) -> None:
        """
        Run joint Re-Pair over all sentences.

        Each iteration:
          1. Count every adjacent pair across the whole corpus.
          2. Replace the most frequent pair with a new non-terminal everywhere.
          3. Record the rule and its frequency.
        Repeats until no pair exceeds min_freq or max_rules is reached.

        More data → more rule reuse → richer non-terminals → better generation.
        """
        print("Tokenising corpus …")
        corpus: list[list[int]] = []
        for s in sentences:
            tokens = s.lower().split()
            if tokens:
                corpus.append([self._get_id(t) for t in tokens])

        total = sum(len(s) for s in corpus)
        print(f"  {len(corpus):,} sentences | {total:,} tokens "
              f"| {len(self.tok2id):,} unique tokens")
        print(f"Running Re-Pair (max_rules={max_rules}, min_freq={min_freq}) …")

        for step in range(max_rules):
            pair_freq: Counter = Counter()
            for seq in corpus:
                for i in range(len(seq) - 1):
                    pair_freq[(seq[i], seq[i + 1])] += 1

            if not pair_freq:
                print(f"  No pairs left after step {step}.")
                break

            best_pair, freq = pair_freq.most_common(1)[0]
            if freq < min_freq:
                print(f"  Max pair frequency dropped below {min_freq} at step {step}.")
                break

            nt = self._new_nt()
            self.rules[nt]     = best_pair
            self.rule_freq[nt] = freq

            new_corpus: list[list[int]] = []
            for seq in corpus:
                new_seq: list[int] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(nt)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus = new_corpus

            if (step + 1) % 100 == 0 or step < 5:
                l, r = best_pair
                print(f"  [{step+1:>4}] NT{nt} → "
                      f"{self.sym_label(l)} + {self.sym_label(r)} "
                      f"(freq={freq:,})")

        self._compressed    = corpus
        self._prefix_index  = None
        print(f"Grammar complete: {len(self.rules):,} rules learned.")

    # ── symbol expansion ──────────────────────────────────────────────────────

    def expand(self, sym: int, _depth: int = 0) -> list[str]:
        """Recursively expand a symbol to a flat list of token strings."""
        if _depth > 300:
            return []
        if sym in self.rules:
            l, r = self.rules[sym]
            return self.expand(l, _depth + 1) + self.expand(r, _depth + 1)
        if sym in self.id2tok:
            return [self.id2tok[sym]]
        return []

    # ── generation ────────────────────────────────────────────────────────────

    def _build_prefix_index(self) -> None:
        """
        Index compressed corpus: symbol → list of symbols that followed it.
        This is the bigram distribution in grammar-symbol space and is the
        core of the autoregressive generation loop.
        """
        print("Building prefix index …")
        idx: dict[int, list[int]] = defaultdict(list)
        for seq in self._compressed:
            for i in range(len(seq) - 1):
                idx[seq[i]].append(seq[i + 1])
        self._prefix_index = dict(idx)
        print(f"  Index covers {len(self._prefix_index):,} symbols.")

    def _weighted_choice(self, candidates: list[int]) -> int:
        freq  = Counter(candidates)
        total = sum(freq.values())
        r     = random.random() * total
        cum   = 0.0
        for sym, count in freq.items():
            cum += count
            if r <= cum:
                return sym
        return candidates[-1]

    def generate(self, prompt: Optional[str] = None,
                 max_symbols: int = 40) -> str:
        """
        Autoregressively generate text.

        1. Encode prompt → starting grammar symbol (last known token).
        2. At each step look up which symbols followed the current one in the
           compressed corpus; pick one weighted by frequency.
        3. Expand each chosen symbol to tokens.
        4. Continue for max_symbols steps or until no continuations exist.

        Longer chains (higher max_symbols) produce longer, richer output as the
        grammar accumulates more structure from larger corpora.
        """
        if self._prefix_index is None:
            self._build_prefix_index()

        # Starting symbol
        if prompt:
            tokens = prompt.lower().split()
            current = None
            for tok in reversed(tokens):
                if tok in self.tok2id:
                    current = self.tok2id[tok]
                    break
            if current is None:
                return f"[no known tokens in prompt: '{prompt}']"
            output = list(tokens)
        else:
            if not self._compressed:
                return "[grammar not trained yet]"
            seq = random.choice(self._compressed)
            if not seq:
                return "[empty sequence in corpus]"
            current = random.choice(seq)
            output  = self.expand(current)

        for _ in range(max_symbols):
            candidates = self._prefix_index.get(current, [])
            if not candidates:
                break
            next_sym = self._weighted_choice(candidates)
            output.extend(self.expand(next_sym))
            current = next_sym

        return " ".join(output) if output else "[no output]"

    # ── inspection ────────────────────────────────────────────────────────────

    def top_rules(self, n: int = 20) -> None:
        print(f"\nTop {n} rules by creation frequency:")
        for nt, freq in self.rule_freq.most_common(n):
            l, r = self.rules[nt]
            print(f"  NT{nt:>6}  ({freq:>6}×)  →  "
                  f"{self.sym_label(l)}  +  {self.sym_label(r)}")

    def stats(self) -> None:
        total = sum(len(s) for s in self._compressed)
        print(f"\nGrammar stats:")
        print(f"  Rules          : {len(self.rules):,}")
        print(f"  Vocabulary     : {len(self.tok2id):,} terminals")
        print(f"  Compressed len : {total:,} symbols")
        print(f"  Next NT id     : {self.next_nt}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Grammar saved → {path}")

    @staticmethod
    def load(path: str) -> "RePairGrammar":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"Grammar loaded ← {path}  ({len(obj.rules):,} rules)")
        return obj
