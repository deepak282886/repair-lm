"""
Corpus-based language environment for stages 4–7.
Uses real text — Alice in Wonderland (free, Project Gutenberg).
"""

import re
import random
from typing import List, Tuple, Set
from environments import BaseEnv


def load_and_clean(path: str) -> List[str]:
    """Load text, clean, tokenize into words."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Strip Gutenberg header/footer
    start = raw.find("CHAPTER I")
    if start == -1:
        start = raw.find("CHAPTER 1")
    if start == -1:
        start = 0
    end = raw.rfind("End of the Project Gutenberg")
    if end == -1:
        end = len(raw)
    raw = raw[start:end]

    # Lowercase, remove punctuation except apostrophes
    raw = raw.lower()
    raw = re.sub(r"[^a-z\s']", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    tokens = raw.split()
    return tokens


def build_sentences(tokens: List[str],
                    min_len: int = 4,
                    max_len: int = 12) -> List[List[str]]:
    """
    Slide a window over tokens to create training sequences.
    Each sequence is min_len to max_len tokens long.
    """
    sentences = []
    step = min_len // 2
    for i in range(0, len(tokens) - max_len, step):
        length = random.randint(min_len, max_len)
        seq = tokens[i:i + length]
        sentences.append(seq)
    return sentences


class CorpusLangEnv(BaseEnv):
    """
    Language environment backed by real text corpus.

    Stage 4: short windows (4-6 tokens), top 500 vocab only
    Stage 5: medium windows (5-9 tokens), top 1000 vocab
    Stage 6: longer windows (6-12 tokens), top 2000 vocab
    Stage 7: full windows (8-16 tokens), open vocab
    """

    STAGE_CONFIG = {
        4: {"min_len": 4,  "max_len": 6,  "vocab_cap": 500},
        5: {"min_len": 5,  "max_len": 9,  "vocab_cap": 1000},
        6: {"min_len": 6,  "max_len": 12, "vocab_cap": 2000},
        7: {"min_len": 8,  "max_len": 16, "vocab_cap": 999999},
    }

    def __init__(self, corpus_path: str, stage: int = 4):
        assert stage in (4, 5, 6, 7)
        self.stage   = stage
        cfg          = self.STAGE_CONFIG[stage]
        self.min_len = cfg["min_len"]
        self.max_len = cfg["max_len"]
        vocab_cap    = cfg["vocab_cap"]

        # Load corpus
        all_tokens = load_and_clean(corpus_path)
        print(f"  [corpus] loaded {len(all_tokens)} tokens from {corpus_path}")

        # Build vocab frequency list
        from collections import Counter
        freq = Counter(all_tokens)
        self.allowed: Set[str] = {
            w for w, _ in freq.most_common(vocab_cap)
        }

        # Filter tokens to allowed vocab
        filtered = [t for t in all_tokens if t in self.allowed]
        print(f"  [corpus] stage {stage}: "
              f"vocab={len(self.allowed)}, "
              f"filtered tokens={len(filtered)}")

        # Build sentence windows
        self._all_seqs = build_sentences(
            filtered, self.min_len, self.max_len
        )
        random.shuffle(self._all_seqs)
        print(f"  [corpus] stage {stage}: "
              f"{len(self._all_seqs)} training sequences")

        self._sentence: List[str] = []
        self._pos: int             = 0
        self._idx: int             = 0   # cursor through sequences

    # ── BaseEnv interface ────────────────────

    def reset(self) -> str:
        self._sentence = self._all_seqs[self._idx % len(self._all_seqs)]
        self._idx     += 1
        self._pos      = 0
        return self._sentence[0]

    def step(self, node_id: str) -> Tuple[float, bool, dict]:
        if not self._sentence or self._pos >= len(self._sentence) - 1:
            return 0.0, True, {}

        expected = self._sentence[self._pos + 1]

        if node_id == expected:
            self._pos += 1
            done = self._pos >= len(self._sentence) - 1
            return 1.0, done, {"match": True}
        elif node_id in self._sentence:
            # partial credit — right domain, wrong position
            return 0.3, False, {"partial": True}
        elif node_id in self.allowed:
            # at least a valid word
            return 0.05, False, {"valid_word": True}
        else:
            return -0.1, False, {"miss": True}

    def token_sequences(self) -> List[List[str]]:
        """Return a batch of sequences for graph seeding."""
        batch_size = 20
        start = (self._idx * batch_size) % max(
            len(self._all_seqs) - batch_size, 1
        )
        return self._all_seqs[start:start + batch_size]

    def intrinsic_reward(self, node_id: str) -> float:
        return 0.02 if node_id in self.allowed else -0.05