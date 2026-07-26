"""
Parallel Training — Federated Graph Merge

Architecture:
  N worker processes each run an independent Graph + Curriculum.
  Every MERGE_EVERY ticks, workers send their graph state to the master.
  Master merges all graphs by averaging edge weights, then broadcasts
  the merged graph back to all workers.
  Workers resume from the merged graph.

Why this is principled (not just a speed hack):
  Multiple graphs exploring independently then merging is analogous to
  parallel collision histories in the Boltzmann derivation.
  The merge is the macroscopic averaging step.
  Each worker has a slightly different epsilon for exploration diversity.

Usage:
  python3 train_parallel.py                    # use all CPU cores
  python3 train_parallel.py --workers 4        # use 4 workers
  python3 train_parallel.py --stages 1-3       # gym stages only
  python3 train_parallel.py --merge-every 200  # merge frequency
  python3 train_parallel.py --chat-only        # load graph.pkl and chat
"""

import os
import sys
import time
import pickle
import argparse
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional

from core import (
    Graph, Node, Edge, NodeKind, EdgeKind, Stage,
    CurriculumScheduler, run_traversal, decay_pass,
    INITIAL_EDGE_WEIGHT, EXPLORATION_GRACE_TICKS,
)
from gym_lang import LangGymEnv
from corpus_lang import CorpusLangEnv
from curriculum_lang import build_language_curriculum
from generator import chat_loop, talk, graph_stats_summary
from environments import BaseEnv

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "alice.txt")
GRAPH_SAVE  = os.path.join(os.path.dirname(__file__), "graph.pkl")


# ─────────────────────────────────────────────
# Graph serialisation helpers
# ─────────────────────────────────────────────

def graph_to_state(graph: Graph) -> dict:
    """Serialise graph to plain dicts for IPC (faster than pickling full graph)."""
    return {
        "nodes": {
            nid: (n.kind.value, n.created_at, n.last_active_at)
            for nid, n in graph.nodes.items()
        },
        "edges": {
            key: (
                e.src, e.dst, e.weight, e.kind.value,
                e.created_at, e.grace_until,
                e.last_reward_at, e.access_count,
            )
            for key, e in graph.edges.items()
        },
        "EMA_floor":        graph.EMA_floor,
        "EMA_edge":         graph.EMA_edge,
        "EMA_len":          graph.EMA_len,
        "EMA_path_success": graph.EMA_path_success,
    }


def state_to_graph(state: dict) -> Graph:
    """Deserialise back to Graph object."""
    g = Graph()
    g.EMA_floor        = state["EMA_floor"]
    g.EMA_edge         = state["EMA_edge"]
    g.EMA_len          = state["EMA_len"]
    g.EMA_path_success = state["EMA_path_success"]

    kind_map = {k.value: k for k in NodeKind}
    for nid, (kv, cat, lat) in state["nodes"].items():
        g.nodes[nid] = Node(
            id=nid, kind=kind_map[kv],
            created_at=cat, last_active_at=lat,
        )

    edge_kind_map = {k.value: k for k in EdgeKind}
    for key, (src, dst, w, kv, cat, gu, lrat, ac) in state["edges"].items():
        g.edges[key] = Edge(
            src=src, dst=dst, weight=w,
            kind=edge_kind_map[kv],
            created_at=cat, grace_until=gu,
            last_reward_at=lrat, access_count=ac,
        )
        g._adj_add(src, dst)   # rebuilds both _adj and _radj
    return g


# ─────────────────────────────────────────────
# Merge strategy
# ─────────────────────────────────────────────

def merge_graphs(states: List[dict], n_workers: int = 0) -> dict:
    """
    Merge N graph states into one.

    Nodes:   union of all nodes across workers.
    Edges:   for each (src,dst) present in any worker:
               weight       = average across workers that have it
               access_count = sum across all workers (total traversals)
               last_reward  = max (most recently rewarded)
               grace_until  = max (most protected)
               kind         = HIGHWAY if any worker has HIGHWAY, else original
    EMAs:    average across all workers.
    """
    if not states:
        return {}
    total = n_workers if n_workers > 0 else len(states)

    # ── Nodes: union ────────────────────────────────────────
    merged_nodes: dict = {}
    for state in states:
        for nid, ndata in state["nodes"].items():
            if nid not in merged_nodes:
                merged_nodes[nid] = ndata
            else:
                # Keep earlier created_at, latest last_active_at
                kv, cat, lat = ndata
                _, ecat, elat = merged_nodes[nid]
                merged_nodes[nid] = (kv, min(cat, ecat), max(lat, elat))

    # ── Edges: weighted average ──────────────────────────────
    edge_accumulator: Dict[tuple, list] = {}
    for state in states:
        for key, edata in state["edges"].items():
            if key not in edge_accumulator:
                edge_accumulator[key] = []
            edge_accumulator[key].append(edata)

    merged_edges: dict = {}
    for key, edata_list in edge_accumulator.items():
        src, dst = key
        weights      = [d[2] for d in edata_list]
        kinds        = [d[3] for d in edata_list]
        created_ats  = [d[4] for d in edata_list]
        grace_untils = [d[5] for d in edata_list]
        last_rewards = [d[6] for d in edata_list]
        access_counts= [d[7] for d in edata_list]

        avg_weight   = sum(weights) / len(weights)
        # Average access counts across workers -- summing inflates
        # the count and makes marginal highways look dominant after merge.
        total_access = sum(access_counts) // len(access_counts)
        max_reward   = max(last_rewards)
        max_grace    = max(grace_untils)
        min_created  = min(created_ats)

        # Store whether any worker had HIGHWAY -- resolved after weight percentile
        any_highway = "HIGHWAY" in kinds

        merged_edges[key] = (
            src, dst, avg_weight,
            "HIGHWAY" if any_highway else kinds[0],  # tentative kind
            min_created, max_grace,
            max_reward, total_access,
        )

    # ── Highway kind: keep only if merged weight >= 90th percentile ───
    # An edge earned highway status through weight -- if averaging across
    # workers diluted it below top-10% of all edges, it reverts to ORDINARY.
    # This is self-regulating: the graph sets its own highway bar.
    all_weights = [v[2] for v in merged_edges.values()]
    if all_weights:
        all_weights_sorted = sorted(all_weights)
        p90_idx  = (0.90 * (len(all_weights_sorted) - 1))
        lo, hi   = int(p90_idx), min(int(p90_idx) + 1, len(all_weights_sorted) - 1)
        frac     = p90_idx - lo
        w90      = all_weights_sorted[lo] * (1 - frac) + all_weights_sorted[hi] * frac
    else:
        w90 = 0.0

    for key, edata in merged_edges.items():
        src, dst, w, kind, cat, gu, lrat, ac = edata
        if kind == "HIGHWAY" and w < w90:
            kind = "ORDINARY"   # diluted below top-10% weight -- reverts
        merged_edges[key] = (src, dst, w, kind, cat, gu, lrat, ac)

    # ── EMAs: average ────────────────────────────────────────
    n = len(states)
    return {
        "nodes": merged_nodes,
        "edges": merged_edges,
        "EMA_floor":        sum(s["EMA_floor"]        for s in states) / n,
        "EMA_edge":         sum(s["EMA_edge"]          for s in states) / n,
        "EMA_len":          sum(s["EMA_len"]           for s in states) / n,
        "EMA_path_success": sum(s["EMA_path_success"]  for s in states) / n,
    }


# ─────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────

def env_for_stage(stage_name: str, corpus_path: str) -> BaseEnv:
    if stage_name.startswith("1_"): return LangGymEnv(stage=1)
    if stage_name.startswith("2_"): return LangGymEnv(stage=2)
    if stage_name.startswith("3_"): return LangGymEnv(stage=3)
    if stage_name.startswith("4_"): return CorpusLangEnv(corpus_path, stage=4)
    if stage_name.startswith("5_"): return CorpusLangEnv(corpus_path, stage=5)
    if stage_name.startswith("6_"): return CorpusLangEnv(corpus_path, stage=6)
    return CorpusLangEnv(corpus_path, stage=7)


# ─────────────────────────────────────────────
# Worker process
# ─────────────────────────────────────────────

def worker_process(
    worker_id:   int,
    stages:      List[Stage],
    corpus_path: str,
    send_q:      mp.Queue,   # worker -> master
    recv_q:      mp.Queue,   # master -> worker
    merge_every: int,
    epsilon_jitter: float,   # small per-worker epsilon variation
    max_ticks:   int,
):
    """
    Independent training loop.
    Runs merge_every ticks, sends state to master, waits for merged state,
    loads it, continues.
    """
    graph      = Graph()
    curriculum = CurriculumScheduler([
        Stage(s.name, s.target_density, s.target_highways,
              # slight epsilon variation per worker for diversity
              max(0.0, s.suggested_epsilon + epsilon_jitter),
              s.min_ticks)
        for s in stages
    ])

    active_env       = None
    active_stage_name = None
    tick             = 0
    decay_every      = 100

    while not curriculum.is_complete() and tick < max_ticks:
        stage   = curriculum.current_stage
        epsilon = curriculum.current_exploration_rate()

        if active_stage_name != stage.name:
            active_env        = env_for_stage(stage.name, corpus_path)
            active_stage_name = stage.name

        # Seed tokens
        if tick % 10 == 0:
            for seq in active_env.token_sequences()[:8]:
                curriculum.feed_tokens(seq, tick, graph)

        # Traversal episode
        start = active_env.reset()
        if start not in graph.nodes:
            graph.nodes[start] = Node(
                id=start, kind=NodeKind.ATOM, created_at=tick
            )

        run_traversal(
            start, tick, epsilon, graph,
            active_env.external_reward,
            active_env.intrinsic_reward,
        )

        if tick % decay_every == 0:
            decay_pass(graph, tick)

        curriculum.step(graph)

        # ── Merge point ──────────────────────────────────────
        if tick > 0 and tick % merge_every == 0:
            send_q.put((worker_id, graph_to_state(graph)))
            merged_state = recv_q.get()   # blocks until master responds
            graph = state_to_graph(merged_state)

        tick += 1

    # Send final state then None sentinel to signal completion
    send_q.put((worker_id, graph_to_state(graph)))
    recv_q.get()   # receive final merged state
    send_q.put((worker_id, None))   # sentinel: worker is done


# ─────────────────────────────────────────────
# Master process
# ─────────────────────────────────────────────

class ParallelTrainer:

    def __init__(self,
                 stages:      List[Stage],
                 corpus_path: str,
                 n_workers:   int,
                 merge_every: int  = 300,
                 max_ticks:   int  = 25000,
                 verbose:     bool = True):
        self.stages      = stages
        self.corpus_path = corpus_path
        self.n_workers   = n_workers
        self.merge_every = merge_every
        self.max_ticks   = max_ticks
        self.verbose     = verbose
        self.graph: Optional[Graph] = None

    def run(self) -> Graph:
        print(f"\n{'='*60}")
        print(f"  Parallel Language Training")
        print(f"  Workers    : {self.n_workers}")
        print(f"  Merge every: {self.merge_every} ticks")
        print(f"  Max ticks  : {self.max_ticks} per worker")
        print(f"{'='*60}\n")

        # One send queue (workers -> master) and per-worker recv queues
        send_q   = mp.Queue()
        recv_qs  = [mp.Queue() for _ in range(self.n_workers)]

        # Epsilon jitter: spread workers slightly around base epsilon
        # Worker 0: -0.05, Worker 1: 0, Worker 2: +0.05, etc.
        jitters = [
            (i - self.n_workers // 2) * 0.05
            for i in range(self.n_workers)
        ]

        # Launch workers
        workers = []
        for wid in range(self.n_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    wid, self.stages, self.corpus_path,
                    send_q, recv_qs[wid],
                    self.merge_every, jitters[wid],
                    self.max_ticks,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        t0       = time.time()
        merge_n  = 0
        done     = set()
        final_states: List[dict] = []

        total_merges = (self.max_ticks // self.merge_every) + 1

        while len(done) < self.n_workers:
            # Collect one state from each active worker
            states_this_round: Dict[int, dict] = {}

            # Wait for all active workers to report
            active = self.n_workers - len(done)
            for _ in range(active):
                wid, state = send_q.get()
                states_this_round[wid] = state

            # Merge (skip None sentinels from finished workers)
            all_states = [s for s in states_this_round.values() if s is not None]
            if all_states:
                merged = merge_graphs(all_states, n_workers=self.n_workers)
            # else: keep last merged state
            merge_n += 1

            # Broadcast merged graph back to still-active workers
            for wid, rq in enumerate(recv_qs):
                if wid not in done and wid in states_this_round:
                    rq.put(merged)

            # Detect finished workers: they send None as their state
            for wid, state in states_this_round.items():
                if state is None:
                    done.add(wid)

            elapsed = time.time() - t0
            merged_graph = state_to_graph(merged)
            stats = merged_graph.stats()

            if self.verbose:
                print(
                    f"  merge={merge_n:3d}/{total_merges} | "
                    f"elapsed={elapsed:5.0f}s | "
                    f"nodes={stats['nodes']:4d} | "
                    f"chunks={stats['chunks']:3d} | "
                    f"highways={stats['highways']:3d} | "
                    f"edges={stats['edges']:4d}"
                )

            # Mid-training chat demo every 5 merges
            if merge_n % 5 == 0:
                self._chat_demo(merged_graph)

        # Final merge of all terminal states
        for p in workers:
            p.join(timeout=5)

        self.graph = state_to_graph(merged)
        self._final_report(time.time() - t0)
        self._save()
        return self.graph

    def _chat_demo(self, graph: Graph):
        prompts = ["the cat", "alice said", "she was", "the queen"]
        print(f"  -- chat demo --")
        for p in prompts:
            r = talk(graph, p, max_tokens=8, temperature=0.3)
            if "(none" not in r and "[unknown]" not in r:
                print(f"    [{p}] -> [{r}]")

    def _final_report(self, elapsed: float):
        if self.graph is None:
            return
        stats = self.graph.stats()
        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed:.1f}s")
        print(f"  {graph_stats_summary(self.graph)}")
        print(f"{'='*60}")

        print(f"\n  Top 10 edges:")
        top = sorted(self.graph.edges.values(),
                     key=lambda e: e.weight, reverse=True)[:10]
        for e in top:
            print(f"    {e.src:20s} -> {e.dst:20s}  "
                  f"w={e.weight:.2f}  {e.kind.value}")

        # Show hierarchy levels
        print(f"\n  Chunk hierarchy:")
        atoms  = [n for n in self.graph.nodes.values() if n.kind == NodeKind.ATOM]
        chunks = [n for n in self.graph.nodes.values() if n.kind == NodeKind.CHUNK]
        l1, l2, l3plus = [], [], []
        for c in chunks:
            members = [
                e.dst for e in self.graph.outgoing_edges(c.id)
                if e.kind == EdgeKind.CHUNK_CONSTITUENT
            ]
            has_chunk_member = any(
                self.graph.nodes.get(m) and
                self.graph.nodes[m].kind == NodeKind.CHUNK
                for m in members
            )
            if not has_chunk_member:
                l1.append(c)
            else:
                # Check if any member is itself a level-2+ chunk
                deep = any(
                    self.graph.nodes.get(m) and
                    self.graph.nodes[m].kind == NodeKind.CHUNK and
                    any(
                        self.graph.nodes.get(e2.dst) and
                        self.graph.nodes[e2.dst].kind == NodeKind.CHUNK
                        for e2 in self.graph.outgoing_edges(m)
                        if e2.kind == EdgeKind.CHUNK_CONSTITUENT
                    )
                    for m in members
                )
                if deep:
                    l3plus.append(c)
                else:
                    l2.append(c)

        print(f"    Level 1 (atom+atom)   : {len(l1)}")
        print(f"    Level 2 (chunk+atom)  : {len(l2)}")
        print(f"    Level 3+ (chunk+chunk): {len(l3plus)}")

    def _save(self):
        if self.graph is None:
            return
        with open(GRAPH_SAVE, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"\n  Graph saved to {GRAPH_SAVE}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_stage_range(s: str) -> List[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a) - 1, int(b)))
    return [int(s) - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",     type=int,   default=mp.cpu_count() - 2,
                        help="Number of parallel workers (default: all cores)")
    parser.add_argument("--merge-every", type=int,   default=300,
                        help="Merge graphs every N ticks (default: 300)")
    parser.add_argument("--max-ticks",   type=int,   default=25000,
                        help="Max ticks per worker (default: 25000)")
    parser.add_argument("--stages",      type=str,   default="1-7",
                        help="Stage range e.g. '1-3' or '4-7'")
    parser.add_argument("--chat-only",   action="store_true",
                        help="Skip training, load saved graph and chat")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Chat generation temperature (default: 0.3)")
    args = parser.parse_args()

    if args.chat_only:
        if not os.path.exists(GRAPH_SAVE):
            print(f"No saved graph at {GRAPH_SAVE}. Train first.")
            sys.exit(1)
        with open(GRAPH_SAVE, "rb") as f:
            graph = pickle.load(f)
        print(f"Loaded: {graph_stats_summary(graph)}")
        chat_loop(graph, temperature=args.temperature)
        sys.exit(0)

    all_stages    = build_language_curriculum()
    stage_indices = parse_stage_range(args.stages)
    stages        = [all_stages[i] for i in stage_indices if i < len(all_stages)]

    if not stages:
        print(f"No valid stages in '{args.stages}'")
        sys.exit(1)

    print(f"Stages: {[s.name for s in stages]}")
    print(f"Workers: {args.workers}")

    trainer = ParallelTrainer(
        stages      = stages,
        corpus_path = CORPUS_PATH,
        n_workers   = args.workers,
        merge_every = args.merge_every,
        max_ticks   = args.max_ticks,
        verbose     = True,
    )

    graph = trainer.run()

    print("\n  Training done. Entering chat mode...\n")
    chat_loop(graph, temperature=args.temperature)