"""
Gym environments for language stages 1–3.
Fully procedural — no real corpus needed.
"""

import random
from typing import List, Tuple
from environments import BaseEnv
from vocab import (
    STAGE1_WORDS, STAGE2_EXTRA, STAGE3_EXTRA,
    STAGE1_TEMPLATES, STAGE2_TEMPLATES, STAGE3_TEMPLATES,
    fill_template,
)


class LangGymEnv(BaseEnv):
    """
    Language gym environment.

    Each episode:
      - Sample a sentence template for the current stage
      - Fill slots with random vocab
      - Start node = first token
      - Reward = +1.0 if traversal reaches correct next token
               = +0.2 if traversal reaches any token in the sentence
               = -0.1 otherwise
    """

    def __init__(self, stage: int = 1):
        assert stage in (1, 2, 3)
        self.stage      = stage
        self._sentence  = []
        self._pos       = 0
        self._build_vocab()

    def _build_vocab(self):
        self.vocab = list(STAGE1_WORDS)
        if self.stage >= 2:
            self.vocab += STAGE2_EXTRA
        if self.stage >= 3:
            self.vocab += STAGE3_EXTRA

    def _templates(self):
        if self.stage == 1: return STAGE1_TEMPLATES
        if self.stage == 2: return STAGE2_TEMPLATES
        return STAGE3_TEMPLATES

    def _sample_sentence(self) -> List[str]:
        tmpl = random.choice(self._templates())
        return fill_template(tmpl)

    # ── BaseEnv interface ────────────────────

    def reset(self) -> str:
        self._sentence = self._sample_sentence()
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
            return 0.2, False, {"partial": True}
        else:
            return -0.1, False, {"miss": True}

    def token_sequences(self) -> List[List[str]]:
        """Generate a fresh batch of sentences as token sequences."""
        seqs = []
        for _ in range(30):   # 30 sentences per batch
            seqs.append(self._sample_sentence())
        return seqs

    def intrinsic_reward(self, node_id: str) -> float:
        """Small bonus for staying on vocabulary."""
        return 0.05 if node_id in self.vocab else -0.05