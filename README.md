# Cognitive Architecture v9.2

A graph-based learning system that builds hierarchical knowledge representations through Hebbian reinforcement — no neural networks, no backpropagation, CPU-native.

---

## What it is

A single weighted graph where nodes are concepts and edges are learned associations. The system trains by traversing the graph, rewarding successful paths, and letting everything else decay. Structure emerges bottom-up:

- **Atoms** — primitive tokens (words, strokes, sensor readings)
- **Chunks** — pairs of atoms that co-activate consistently, promoted into higher-level nodes
- **Highways** — long-range shortcuts between frequently-rewarded path endpoints

The same update rule applies at every level. Chunks of chunks form naturally. No special-casing per hierarchy level.

---

## Structural analogy

The architecture shares its conceptual core with the resolution of Hilbert's 6th problem (Deng-Hani, building on Lanford): just as the Boltzmann equation emerges by focusing only on well-behaved collision histories and discarding pathological ones, this system focuses only on well-rewarded traversal paths — decay prunes the rest.

| Boltzmann / Hilbert 6th | This Architecture |
|---|---|
| Collision history trees | Traversal path histories |
| Recollisions → measure zero | Unrewarded paths → decay to zero |
| Molecular chaos assumption | Local EMA (neighborhood statistic) |
| Coarse-grained Boltzmann equation | Chunk nodes (compressed abstractions) |
| Long-range correlations → negligible | Highway formation (long-range shortcuts) |
| Collision kernel | Reward function |

---

## Key design decisions

**Everything is self-regulating — no fixed thresholds.**

- **Chunk promotion** fires when an edge's access count reaches the 90th percentile of all edge traversals in the graph. The graph sets its own bar.
- **Highway formation** fires when a path's accumulated success reaches the 90th percentile of all path successes seen. Same principle.
- **Highway capacity** is bounded by `sqrt(node_count) * 2`. New stronger highways displace weaker ones. Scales sub-linearly with vocabulary.
- **Decay rate** adapts to graph density: current edges-per-node divided by the historical EMA of edges-per-node. Dense → aggressive pruning. Sparse → gentle.
- **Minimum path length** for highway formation grows with average chunk hierarchy depth. Highways compress meaningful sequences, not trivial bigrams.
- **Coarse-to-fine search** — traversal consults highways first, then chunk-level index edges, then ordinary atom edges. Falls through to the next level if a choice leads to a dead end.

**Curriculum is a read-only observer.**

The curriculum has zero write access to graph weights. It observes graph statistics and decides what exercises to present and what exploration rate to use. The graph and the curriculum are strictly separated.

**Chunk promotion requires same-level pairs.**

Atoms promote with atoms → L1 chunks. L1 chunks promote with L1 chunks → L2 chunks. This prevents newly created chunks from immediately cascading into higher-level chunks before the atom layer stabilises.

---

## Files

```
core.py            — graph, Hebbian update, chunk promotion, highway formation,
                     adaptive decay, coarse-to-fine traversal, curriculum scheduler
environments.py    — BaseEnv interface + WordChain and NumberSequence demo envs
vocab.py           — word lists and sentence templates per gym stage
gym_lang.py        — procedural language gym (stages 1-3)
corpus_lang.py     — real text corpus environment (stages 4-7)
curriculum_lang.py — 7-stage language curriculum definitions
generator.py       — graph traversal → text generation + interactive chat loop
train_language.py  — single-process training script
train_parallel.py  — parallel training with federated graph merge
```

---

## Seven-stage language curriculum

| Stage | Name | Vocab | Window | Notes |
|---|---|---|---|---|
| 1 | atoms | ~40 | 2-4 tok | Seed atom layer, high exploration |
| 2 | bigrams | ~65 | 4-6 tok | Chunk formation begins |
| 3 | sentences | ~85 | 5-8 tok | SVO patterns, highways start |
| 4 | real_short | 500 | 4-6 tok | Real corpus co-occurrence enters |
| 5 | real_medium | 1000 | 5-9 tok | Highway consolidation |
| 6 | real_long | 2000 | 6-12 tok | Vocabulary saturation |
| 7 | open | full | 8-16 tok | Consolidation + conversation |

Stages 1-3 use a procedural gym with synthetic sentences. Stages 4-7 use real text (Alice in Wonderland by default — free, Project Gutenberg).

---

## Quickstart

```bash
# Download corpus
curl -o alice.txt https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt

# Single-process training (all 7 stages, then chat)
python3 train_language.py

# Parallel training (uses all CPU cores)
python3 train_parallel.py

# Specific stages only
python3 train_language.py --stages 1-3

# Load saved graph and chat
python3 train_language.py --chat-only

# Adjust generation temperature
python3 train_language.py --chat-only --temperature 0.5
```

In chat mode: type `stats` for graph state, `temp 0.5` to adjust temperature.

The graph is saved to `graph.pkl` after training.

---

## Parallel training

```bash
python3 train_parallel.py --workers 8 --merge-every 300 --stages 1-7
```

Each worker runs an independent graph with a slightly different exploration rate. Every `merge-every` ticks, workers send their graph state to the master, which merges by averaging edge weights and broadcasting back. Highway status in the merged graph is resolved by weight percentile — an edge keeps HIGHWAY only if its averaged weight is in the top 10% of all merged edges.

This mirrors the Boltzmann derivation structurally: parallel collision histories summed into the macroscopic equation.

---

## Sample output (stage 5, Alice corpus)

```
[alice said] → was against it ball can is very old
[she was]    → noticed that done ' said the small cat
[the queen]  → hot next moment children round gloves begin at
```

"noticed that done ' said the small cat" — real Alice dialogue syntax, apostrophe preserved from corpus tokenisation. "children round gloves begin at" — genuine Alice vocabulary in contextually coherent order.

---

## Open questions

- **Reward function** — the language reward (next-token match) is a weak proxy for conversation quality. The architecture is fully implemented; what it learns is entirely determined by the reward signal.
- **Compositional generalization** — predicted to emerge from chunk + highway interaction. Not yet empirically verified on held-out combinations.
- **Self-learning** — when does the system generate its own training signal? Unspecified.
- **Drawing domain** — stroke sequences as atoms, pixel similarity as reward. CPU-native. The hierarchy that emerges would be directly visible.
- **Game domain** — Pokémon Red or TextWorld. Natural stage progression (gym badges = curriculum stages).

---

## Requirements

```
python >= 3.10
numpy
```

No GPU. No deep learning framework. Runs on CPU.

---

## Background

Architecture designed independently. The Boltzmann analogy was identified post-hoc during implementation review.

The pseudocode specification (v9.2) is the authoritative design document. This codebase is a faithful implementation with the following deviations discovered during implementation:

- Chunk promotion abs condition uses access count (traversal frequency) rather than weight relative to local EMA — weight-based comparison fails when a node has a single dominant outgoing edge, as that edge IS its own local average
- Local EMA in `step_continuation` uses global EMA as a fast approximation — full BFS local EMA is O(E) and was a dominant cost at scale; reserved for chunk promotion decisions where precision matters
- `min_ticks` added to curriculum stages — token seeding at stage start would otherwise satisfy readiness thresholds before any traversal occurred