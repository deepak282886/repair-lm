"""
Language training — full 7-stage curriculum.

Usage:
    python3 train_language.py              # train all stages then chat
    python3 train_language.py --chat-only  # load saved graph and chat
    python3 train_language.py --stages 1-3 # train only stages 1-3

The graph is saved to graph.pkl after training so you can chat later
without retraining.
"""

import sys
import time
import pickle
import argparse
import os
from typing import List

from core import (
    Graph, CurriculumScheduler, Stage,
    run_traversal, decay_pass, NodeKind,
)
from gym_lang import LangGymEnv
from corpus_lang import CorpusLangEnv
from curriculum_lang import build_language_curriculum
from generator import chat_loop, talk, graph_stats_summary
from environments import BaseEnv

CORPUS_PATH  = os.path.join(os.path.dirname(__file__), "alice.txt")
GRAPH_SAVE   = os.path.join(os.path.dirname(__file__), "graph.pkl")
DECAY_EVERY  = 100
LOG_EVERY    = 200


# ─────────────────────────────────────────────
# Environment factory — maps stage name to env
# ─────────────────────────────────────────────

def env_for_stage(stage: Stage, corpus_path: str) -> BaseEnv:
    name = stage.name
    if name.startswith("1_"):
        return LangGymEnv(stage=1)
    elif name.startswith("2_"):
        return LangGymEnv(stage=2)
    elif name.startswith("3_"):
        return LangGymEnv(stage=3)
    elif name.startswith("4_"):
        return CorpusLangEnv(corpus_path, stage=4)
    elif name.startswith("5_"):
        return CorpusLangEnv(corpus_path, stage=5)
    elif name.startswith("6_"):
        return CorpusLangEnv(corpus_path, stage=6)
    else:
        return CorpusLangEnv(corpus_path, stage=7)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class LanguageTrainer:

    def __init__(self, stages: List[Stage],
                 corpus_path: str,
                 verbose: bool = True):
        self.stages      = stages
        self.corpus_path = corpus_path
        self.verbose     = verbose
        self.graph       = Graph()
        self.curriculum  = CurriculumScheduler(stages)
        self.tick        = 0
        self.rewards_log = []
        self._active_env = None

    def _get_env(self) -> BaseEnv:
        stage = self.curriculum.current_stage
        if self._active_env is None or \
           not hasattr(self._active_env, '_stage_name') or \
           self._active_env._stage_name != stage.name:
            print(f"\n  [trainer] Loading env for stage: {stage.name}")
            self._active_env = env_for_stage(stage, self.corpus_path)
            self._active_env._stage_name = stage.name
        return self._active_env

    def run(self):
        print(f"\n{'='*60}")
        print(f"  Language Training — 7-Stage Curriculum")
        print(f"  Corpus: {self.corpus_path}")
        print(f"{'='*60}\n")

        t0 = time.time()

        while not self.curriculum.is_complete():
            env     = self._get_env()
            stage   = self.curriculum.current_stage
            epsilon = self.curriculum.current_exploration_rate()

            # Seed graph with token sequences every 20 ticks
            if self.tick % 20 == 0:
                for seq in env.token_sequences():
                    self.curriculum.feed_tokens(seq, self.tick, self.graph)

            # Run one traversal episode
            start = env.reset()
            if start not in self.graph.nodes:
                from core import Node
                self.graph.nodes[start] = Node(
                    id=start, kind=NodeKind.ATOM, created_at=self.tick
                )

            path = run_traversal(
                start_node         = start,
                tick               = self.tick,
                epsilon            = epsilon,
                graph              = self.graph,
                external_reward_fn = env.external_reward,
                intrinsic_reward_fn= env.intrinsic_reward,
            )

            self.rewards_log.append(path.total_reward)

            # Decay pass
            if self.tick % DECAY_EVERY == 0:
                decay_pass(self.graph, self.tick)

            # Curriculum step
            prev_stage = self.curriculum.current
            self.curriculum.step(self.graph)
            if self.curriculum.current != prev_stage:
                # Stage changed — reset env
                self._active_env = None
                self._mid_stage_chat()

            # Logging
            if self.verbose and self.tick % LOG_EVERY == 0:
                self._log(stage, epsilon, path)

            self.tick += 1

        elapsed = time.time() - t0
        self._final_report(elapsed)
        self._save_graph()

    def _log(self, stage, epsilon, path):
        stats   = self.graph.stats()
        avg_r   = (sum(self.rewards_log[-100:]) /
                   max(len(self.rewards_log[-100:]), 1))
        ticks_in = self.curriculum.ticks_in_stage
        print(
            f"  tick={self.tick:5d} | "
            f"stage={stage.name:18s} | "
            f"ε={epsilon:.2f} | "
            f"avg_r={avg_r:6.3f} | "
            f"nodes={stats['nodes']:4d} | "
            f"chunks={stats['chunks']:3d} | "
            f"highways={stats['highways']:3d} | "
            f"path={len(path.nodes):2d} | "
            f"t_in={ticks_in:4d}/{stage.min_ticks}"
        )

    def _mid_stage_chat(self):
        """Quick demo after each stage advance."""
        print(f"\n  ── Mid-training chat demo ──")
        test_prompts = ["the cat", "a bird", "the sun", "the dog"]
        for p in test_prompts:
            r = talk(self.graph, p, max_tokens=8, temperature=0.3)
            if "[unknown]" not in r and "(none" not in r:
                print(f"    '{p}' → '{r}'")
        print()

    def _final_report(self, elapsed: float):
        stats = self.graph.stats()
        print(f"\n{'='*60}")
        print(f"  Training complete — {self.tick} ticks in {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"  {graph_stats_summary(self.graph)}")
        print(f"\n  Top 10 edges:")
        top = sorted(self.graph.edges.values(),
                     key=lambda e: e.weight, reverse=True)[:10]
        for e in top:
            print(f"    {e.src:15s} -> {e.dst:15s}  "
                  f"w={e.weight:8.2f}  {e.kind.value}")
        print(f"\n  Sample chunks:")
        chunks = [n for n in self.graph.nodes.values()
                  if n.kind == NodeKind.CHUNK][:15]
        for c in chunks:
            members = [e.dst for e in self.graph.outgoing_edges(c.id)
                       if e.kind.value == "CHUNK_CONSTITUENT"]
            print(f"    {' + '.join(members)}")

    def _save_graph(self):
        with open(GRAPH_SAVE, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"\n  Graph saved to {GRAPH_SAVE}")
        print(f"  Load with: graph = pickle.load(open('{GRAPH_SAVE}','rb'))")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_stage_range(s: str) -> List[int]:
    """Parse '1-3' or '4-7' into list of stage indices."""
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a) - 1, int(b)))
    return [int(s) - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-only", action="store_true",
                        help="Skip training, load saved graph and chat")
    parser.add_argument("--stages", type=str, default="1-7",
                        help="Stage range to train, e.g. '1-3' or '4-7'")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature for chat (default 0.3)")
    args = parser.parse_args()

    all_stages = build_language_curriculum()

    if args.chat_only:
        if not os.path.exists(GRAPH_SAVE):
            print(f"No saved graph found at {GRAPH_SAVE}.")
            print("Run training first: python3 train_language.py")
            sys.exit(1)
        print(f"Loading saved graph from {GRAPH_SAVE}...")
        with open(GRAPH_SAVE, "rb") as f:
            graph = pickle.load(f)
        print(f"  {graph_stats_summary(graph)}")
        chat_loop(graph, temperature=args.temperature)
        sys.exit(0)

    # Select stage range
    stage_indices = parse_stage_range(args.stages)
    stages = [all_stages[i] for i in stage_indices
              if i < len(all_stages)]

    if not stages:
        print(f"No valid stages in range '{args.stages}'")
        sys.exit(1)

    print(f"Training stages: {[s.name for s in stages]}")

    trainer = LanguageTrainer(
        stages       = stages,
        corpus_path  = CORPUS_PATH,
        verbose      = True,
    )
    trainer.run()

    # After training — jump straight into chat
    print("\n  Training done. Entering chat mode...\n")
    chat_loop(trainer.graph, temperature=args.temperature)