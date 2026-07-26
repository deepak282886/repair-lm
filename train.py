"""
Main training loop — Section 10 of pseudocode v9.2
Ties Graph + CurriculumScheduler + Environment together.
"""

import time
from typing import List

from core import (
    Graph, CurriculumScheduler, Stage,
    run_traversal, decay_pass, NodeKind, EdgeKind,
)
from environments import BaseEnv, WordChainEnv, NumberSequenceEnv


# ─────────────────────────────────────────────
# Curriculum stage plans
# ─────────────────────────────────────────────

def word_chain_stages() -> List[Stage]:
    return [
        # Stage 1: must form at least 4 chunks and 3 highways before advancing
        # epsilon high — lots of exploration to seed the graph
        # Stage 1: 5 pairs -> expect 2 chunks, 2 highways min
        Stage(name="word_chain_stage1", target_density=0.12,
              target_highways=2,  suggested_epsilon=0.50, min_ticks=500),
        # Stage 2: 10 pairs -> expect ~8 chunks, 4 highways
        Stage(name="word_chain_stage2", target_density=0.30,
              target_highways=4,  suggested_epsilon=0.30, min_ticks=700),
        # Stage 3: 15 pairs -> expect ~18 chunks, 8 highways
        Stage(name="word_chain_stage3", target_density=0.50,
              target_highways=8,  suggested_epsilon=0.15, min_ticks=1000),
    ]


def number_seq_stages() -> List[Stage]:
    return [
        Stage(name="number_stage1", target_density=0.20,
              target_highways=2,  suggested_epsilon=0.50, min_ticks=500),
        Stage(name="number_stage2", target_density=0.40,
              target_highways=5,  suggested_epsilon=0.30, min_ticks=700),
        Stage(name="number_stage3", target_density=0.60,
              target_highways=10, suggested_epsilon=0.15, min_ticks=1000),
    ]


# ─────────────────────────────────────────────
# Training runner
# ─────────────────────────────────────────────

class Trainer:

    def __init__(self, env: BaseEnv, stages: List[Stage],
                 max_ticks: int = 2000, decay_every: int = 50,
                 verbose: bool = True):
        self.env         = env
        self.graph       = Graph()
        self.curriculum  = CurriculumScheduler(stages)
        self.max_ticks   = max_ticks
        self.decay_every = decay_every
        self.verbose     = verbose

        self.tick        = 0
        self.rewards_log = []
        self.stage_log   = []

    def _env_for_stage(self) -> BaseEnv:
        """Return env configured for current curriculum stage."""
        stage_name = self.curriculum.current_stage.name
        if "stage1" in stage_name:
            return self.env.__class__("stage1")
        elif "stage2" in stage_name:
            return self.env.__class__("stage2")
        else:
            return self.env.__class__("stage3")

    def run(self):
        print(f"\n{'='*55}")
        print(f"  Cognitive Architecture Training — v9.2")
        print(f"  Environment : {self.env.__class__.__name__}")
        print(f"  Max ticks   : {self.max_ticks}")
        print(f"{'='*55}\n")

        while (not self.curriculum.is_complete()
               and self.tick < self.max_ticks):

            active_env = self._env_for_stage()
            stage      = self.curriculum.current_stage
            epsilon    = self.curriculum.current_exploration_rate()

            # Feed token sequences every 10 ticks — keeps co-occurrence signal
            # fresh and gives chunk promotion ongoing opportunities
            if self.tick % 10 == 0:
                for seq in active_env.token_sequences():
                    self.curriculum.feed_tokens(seq, self.tick, self.graph)

            # Run traversal (Sections 4/6/7)
            start = active_env.reset()
            if start not in self.graph.nodes:
                import core as _c
                self.graph.nodes[start] = _c.Node(
                    id=start, kind=NodeKind.ATOM, created_at=self.tick)

            path = run_traversal(
                start_node=start,
                tick=self.tick,
                epsilon=epsilon,
                graph=self.graph,
                external_reward_fn=active_env.external_reward,
                intrinsic_reward_fn=active_env.intrinsic_reward,
            )

            self.rewards_log.append(path.total_reward)
            self.stage_log.append(stage.name)

            # Decay pass every N ticks (Section 8)
            if self.tick % self.decay_every == 0:
                decay_pass(self.graph, self.tick)

            # Curriculum step (Section 9) — read-only observer
            self.curriculum.step(self.graph)

            # Logging
            if self.verbose and self.tick % 100 == 0:
                self._log(path, epsilon)

            self.tick += 1

        self._final_report()

    def _log(self, path, epsilon: float):
        stats = self.graph.stats()
        avg_r = (sum(self.rewards_log[-50:]) /
                 max(len(self.rewards_log[-50:]), 1))
        stage = (self.curriculum.current_stage.name
                 if not self.curriculum.is_complete() else "COMPLETE")
        print(f"  tick={self.tick:4d} | stage={stage:25s} | "
              f"ε={epsilon:.2f} | "
              f"avg_r={avg_r:.3f} | "
              f"nodes={stats['nodes']:3d} | "
              f"edges={stats['edges']:3d} | "
              f"chunks={stats['chunks']:2d} | "
              f"highways={stats['highways']:2d} | "
              f"path_len={len(path.nodes):2d}")

    def _final_report(self):
        stats = self.graph.stats()
        print(f"\n{'='*55}")
        print(f"  Training complete — {self.tick} ticks")
        print(f"{'='*55}")
        print(f"  Graph nodes    : {stats['nodes']}")
        print(f"  Graph edges    : {stats['edges']}")
        print(f"  Atoms          : {stats['atoms']}")
        print(f"  Chunks formed  : {stats['chunks']}")
        print(f"  Highways formed: {stats['highways']}")
        print(f"  Final EMA_edge : {stats['EMA_edge']}")
        print(f"\n  Top edges by weight:")
        top = sorted(self.graph.edges.values(),
                     key=lambda e: e.weight, reverse=True)[:10]
        for e in top:
            print(f"    {e.src:20s} -> {e.dst:20s}  "
                  f"w={e.weight:.4f}  kind={e.kind.value}")
        print(f"\n  Chunks:")
        for n in self.graph.nodes.values():
            if n.kind == NodeKind.CHUNK:
                members = [e.dst for e in self.graph.outgoing_edges(n.id)
                           if e.kind == EdgeKind.CHUNK_CONSTITUENT]
                print(f"    {n.id:30s}  members={members}")
        print(f"\n  Highways:")
        for e in self.graph.edges.values():
            if e.kind == EdgeKind.HIGHWAY:
                print(f"    {e.src:20s} -> {e.dst:20s}  w={e.weight:.4f}")
        print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    env_name = sys.argv[1] if len(sys.argv) > 1 else "word_chain"

    if env_name == "word_chain":
        env    = WordChainEnv("stage1")
        stages = word_chain_stages()
    elif env_name == "number_sequence":
        env    = NumberSequenceEnv("stage1")
        stages = number_seq_stages()
    else:
        print(f"Unknown env: {env_name}. Options: word_chain, number_sequence")
        sys.exit(1)

    trainer = Trainer(env=env, stages=stages, max_ticks=8000,
                      decay_every=50, verbose=True)
    t0 = time.time()
    trainer.run()
    print(f"  Wall time: {time.time() - t0:.1f}s")