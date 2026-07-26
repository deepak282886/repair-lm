"""
Environment Interface + Demo Environments

To plug in a new environment, subclass BaseEnv and implement:
  - reset()       -> start_node (str)
  - step(node_id) -> (reward, done, info)
  - token_sequences() -> List[List[str]]   (training sequences for this stage)
"""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────
# Base Interface
# ─────────────────────────────────────────────

class BaseEnv(ABC):

    @abstractmethod
    def reset(self) -> str:
        """Return start node id."""
        ...

    @abstractmethod
    def step(self, node_id: str) -> Tuple[float, bool, dict]:
        """Return (reward, done, info)."""
        ...

    @abstractmethod
    def token_sequences(self) -> List[List[str]]:
        """Training sequences (curriculum exercises) for this env."""
        ...

    def external_reward(self, node_id: str) -> float:
        reward, _, _ = self.step(node_id)
        return reward

    def intrinsic_reward(self, node_id: str) -> float:
        """Override for curiosity/novelty bonus. Default: 0."""
        return 0.0


# ─────────────────────────────────────────────
# Demo Environment 1: Word Chain
#
# Nodes = words. Goal = reach a target word by
# traversing semantically related words.
# Natural curriculum: short chains -> long chains
# -> novel combinations never directly trained.
# ─────────────────────────────────────────────

WORD_CHAINS = {
    # Stage 1 — simple 2-hop chains
    "stage1": [
        ["cat",   "animal", "dog"],
        ["red",   "color",  "blue"],
        ["sun",   "bright", "moon"],
        ["fish",  "swim",   "water"],
        ["bird",  "fly",    "sky"],
    ],
    # Stage 2 — 3-hop chains
    "stage2": [
        ["cat",   "animal", "wild",  "forest"],
        ["sun",   "bright", "light", "lamp"],
        ["fish",  "swim",   "ocean", "wave"],
        ["bird",  "fly",    "sky",   "cloud"],
        ["dog",   "animal", "fur",   "warm"],
    ],
    # Stage 3 — cross-domain, longer
    "stage3": [
        ["cat",   "animal", "wild",  "forest", "tree",  "green"],
        ["sun",   "bright", "light", "lamp",   "dark",  "night"],
        ["fish",  "swim",   "ocean", "wave",   "beach", "sand"],
        ["bird",  "fly",    "sky",   "cloud",  "rain",  "water"],
        ["dog",   "run",    "fast",  "wind",   "air",   "breath"],
    ],
}

TARGET_WORDS = {
    "stage1": {"dog", "blue", "moon", "water", "sky"},
    "stage2": {"forest", "lamp", "wave", "cloud", "warm"},
    "stage3": {"green", "night", "sand", "water", "breath"},
}


class WordChainEnv(BaseEnv):
    """
    Text-based environment: traverse word graph to reach target.
    Reward = +1.0 on reaching target, 0 otherwise.
    CPU-only, no dependencies beyond stdlib.
    """

    def __init__(self, stage: str = "stage1"):
        self.stage       = stage
        self.targets     = TARGET_WORDS[stage]
        self.sequences   = WORD_CHAINS[stage]
        self._current    = None

    def reset(self) -> str:
        seq = random.choice(self.sequences)
        self._current = seq[0]
        self._targets = TARGET_WORDS[self.stage]
        return self._current

    def step(self, node_id: str) -> Tuple[float, bool, dict]:
        if node_id in self._targets:
            return 1.0, True, {"reached": node_id}
        return 0.0, False, {}

    def token_sequences(self) -> List[List[str]]:
        return self.sequences

    def intrinsic_reward(self, node_id: str) -> float:
        """Small novelty bonus for visiting new nodes."""
        return 0.05


# ─────────────────────────────────────────────
# Demo Environment 2: Number Sequence
#
# Nodes = digit strings. Goal = complete arithmetic
# sequences. Tests whether the architecture learns
# numeric "chunks" (e.g., even numbers, multiples).
# ─────────────────────────────────────────────

class NumberSequenceEnv(BaseEnv):
    """
    Traverse number sequences. Reward for reaching correct next number.
    Stage 1: count by 1s. Stage 2: count by 2s. Stage 3: mixed.
    """

    SEQUENCES = {
        "stage1": [
            [str(i) for i in range(1, 6)],   # 1,2,3,4,5
            [str(i) for i in range(2, 7)],
            [str(i) for i in range(3, 8)],
        ],
        "stage2": [
            [str(i) for i in range(0, 11, 2)],  # evens
            [str(i) for i in range(1, 12, 2)],  # odds
            [str(i) for i in range(0, 16, 5)],  # 5s
        ],
        "stage3": [
            [str(i) for i in range(1, 10)],
            [str(i**2) for i in range(1, 7)],   # squares
            [str(i) for i in [1, 1, 2, 3, 5, 8, 13]],  # fibonacci
        ],
    }

    def __init__(self, stage: str = "stage1"):
        self.stage     = stage
        self.sequences = self.SEQUENCES[stage]
        self._goal     = None

    def reset(self) -> str:
        seq = random.choice(self.sequences)
        self._seq  = seq
        self._pos  = 0
        self._goal = seq[-1]
        return seq[0]

    def step(self, node_id: str) -> Tuple[float, bool, dict]:
        if node_id == self._goal:
            return 1.0, True, {"reached": node_id}
        # partial reward for staying on sequence
        if node_id in self._seq:
            return 0.1, False, {"on_sequence": True}
        return -0.05, False, {}

    def token_sequences(self) -> List[List[str]]:
        return self.sequences


# ─────────────────────────────────────────────
# Environment registry — add new envs here
# ─────────────────────────────────────────────

ENVIRONMENTS: Dict[str, type] = {
    "word_chain":      WordChainEnv,
    "number_sequence": NumberSequenceEnv,
}