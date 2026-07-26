"""
Cognitive Architecture — Python Implementation
Based on pseudocode specification v9.2

Changes from initial implementation:
- Chunk promotion uses 90th percentile of edge access_counts (tunable)
- Chunk promotion called inside traversal loop on every consecutive node pair
- Chunk nodes can participate as src/dst in higher-level chunk promotion
- Circular promotion guard: a chunk cannot promote with its own constituents
- All four levels of hierarchy are now reachable: atom->chunk->chunk2->chunk3
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set


# ─────────────────────────────────────────────
# Constants (Section 11)
# ─────────────────────────────────────────────

DECAY                      = 0.85
EMA_ALPHA                  = 0.08
DECAY_STEP                 = 0.06
PROTECT                    = 0.7
PROTECT_WINDOW             = 50
M3_SEED_WEIGHT             = 0.11
ABS_FLOOR_MULT             = 0.55
KAPPA                      = 0.12
TARGET_LEN                 = 3.0
LENGTH_GAIN                = 0.12
SOFT_REJECT                = 0.05
CHUNK_DOM_FRAC             = 0.35
CHUNK_PROMOTION_PERCENTILE = 90.0   # tunable: percentile of access_counts required to promote
CHUNK_MIN_ACCESS           = 20     # hard floor: minimum traversals regardless of percentile
HIGHWAY_MULT               = 1.2
INITIAL_EDGE_WEIGHT        = 0.10
HEBBIAN_GAIN               = 0.15
NEAR_ZERO                  = 0.01
MAX_PATH_LEN               = 20
FRACTION_OF_SUCCESS        = 0.3
EXPLORATION_GRACE_TICKS    = 10
MIN_MASS_FOR_LOCAL_STAT    = 3.0
EPSILON_CUTOFF             = 0.01


# ─────────────────────────────────────────────
# Section 1: Data Structures
# ─────────────────────────────────────────────

class NodeKind(Enum):
    ATOM             = "ATOM"
    CHUNK            = "CHUNK"
    HIGHWAY_ENDPOINT = "HIGHWAY_ENDPOINT"


class EdgeKind(Enum):
    ORDINARY          = "ORDINARY"
    CHUNK_CONSTITUENT = "CHUNK_CONSTITUENT"
    HIGHWAY           = "HIGHWAY"
    INDEX             = "INDEX"


@dataclass
class Node:
    id: str
    kind: NodeKind
    created_at: int
    last_active_at: int = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id


@dataclass
class Edge:
    src: str
    dst: str
    weight: float
    kind: EdgeKind
    created_at: int
    grace_until: int
    last_reward_at: int = -1
    access_count: int = 0

    def key(self):
        return (self.src, self.dst)


@dataclass
class Path:
    nodes: List[str] = field(default_factory=list)
    path_activation: float = 0.0
    total_reward: float = 0.0


class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        # Forward adjacency index: node_id -> set of dst node_ids
        self._adj: Dict[str, Set[str]] = {}
        # Reverse adjacency index: node_id -> set of src node_ids
        self._radj: Dict[str, Set[str]] = {}
        # Density history EMA: tracks edges-per-node over time
        # Used for adaptive decay -- no fixed threshold
        self.EMA_density: float = 1.0

        # Global EMA statistics
        self.EMA_floor: float        = 0.1
        self.EMA_edge: float         = INITIAL_EDGE_WEIGHT
        self.EMA_len: float          = TARGET_LEN
        self.EMA_path_success: float = 0.0

    # ── Adjacency index maintenance ───────────────────────────

    def _adj_add(self, src: str, dst: str):
        if src not in self._adj:
            self._adj[src] = set()
        self._adj[src].add(dst)
        if dst not in self._radj:
            self._radj[dst] = set()
        self._radj[dst].add(src)

    def _adj_remove(self, src: str, dst: str):
        if src in self._adj:
            self._adj[src].discard(dst)
            if not self._adj[src]:
                del self._adj[src]
        if dst in self._radj:
            self._radj[dst].discard(src)
            if not self._radj[dst]:
                del self._radj[dst]

    # ── Helpers ───────────────────────────────────────────────

    def outgoing_edges(self, node_id: str) -> List[Edge]:
        """O(degree) lookup via adjacency index."""
        dsts = self._adj.get(node_id)
        if not dsts:
            return []
        return [self.edges[(node_id, d)] for d in dsts
                if (node_id, d) in self.edges]

    def incoming_edges(self, node_id: str) -> List[Edge]:
        """O(degree) lookup via reverse adjacency index."""
        srcs = self._radj.get(node_id)
        if not srcs:
            return []
        return [self.edges[(s, node_id)] for s in srcs
                if (s, node_id) in self.edges]

    def edges_touching(self, node_id: str) -> List[Edge]:
        """O(degree) via both adjacency indexes."""
        return self.outgoing_edges(node_id) + self.incoming_edges(node_id)

    def other(self, edge: Edge, node_id: str) -> str:
        return edge.dst if edge.src == node_id else edge.src

    def has_no_edges(self, node_id: str) -> bool:
        return (not self._adj.get(node_id)
                and not self._radj.get(node_id))

    def all_reachable_nodes(self, start_id: str) -> List[str]:
        visited: Set[str] = set()
        frontier = [start_id]
        while frontier:
            n = frontier.pop()
            if n in visited:
                continue
            visited.add(n)
            for e in self.outgoing_edges(n):
                if e.dst not in visited:
                    frontier.append(e.dst)
        return list(visited)

    def count_nodes(self, kind: NodeKind) -> int:
        return sum(1 for n in self.nodes.values() if n.kind == kind)

    def count_edges(self, kind: EdgeKind) -> int:
        return sum(1 for e in self.edges.values() if e.kind == kind)

    def stats(self) -> dict:
        return {
            "nodes":    len(self.nodes),
            "edges":    len(self.edges),
            "atoms":    self.count_nodes(NodeKind.ATOM),
            "chunks":   self.count_nodes(NodeKind.CHUNK),
            "highways": self.count_edges(EdgeKind.HIGHWAY),
            "EMA_edge": round(self.EMA_edge, 4),
        }


# ─────────────────────────────────────────────
# Section 2: Local (decay-weighted) EMA
# ─────────────────────────────────────────────

def local_EMA_edge(node_id: str, graph: Graph) -> float:
    total_weight = 0.0
    total_mass   = 0.0
    frontier     = [(node_id, 0)]
    visited: Set[str] = {node_id}

    while frontier:
        n, hop = frontier.pop(0)
        contribution = DECAY ** hop
        if contribution < EPSILON_CUTOFF:
            continue
        for e in graph.edges_touching(n):
            total_weight += e.weight * contribution
            total_mass   += contribution
            other = graph.other(e, n)
            if other not in visited:
                visited.add(other)
                frontier.append((other, hop + 1))

    if total_mass < MIN_MASS_FOR_LOCAL_STAT:
        return graph.EMA_edge
    return total_weight / total_mass


def local_EMA_len(node_id: str, graph: Graph) -> float:
    total_len  = 0.0
    total_mass = 0.0
    frontier   = [(node_id, 0)]
    visited: Set[str] = {node_id}

    while frontier:
        n, hop = frontier.pop(0)
        contribution = DECAY ** hop
        if contribution < EPSILON_CUTOFF:
            continue
        for e in graph.edges_touching(n):
            total_len  += hop * contribution
            total_mass += contribution
            other = graph.other(e, n)
            if other not in visited:
                visited.add(other)
                frontier.append((other, hop + 1))

    if total_mass < MIN_MASS_FOR_LOCAL_STAT:
        return graph.EMA_len
    return total_len / total_mass


def local_EMA_path_success(node_id: str, graph: Graph) -> float:
    total_success = 0.0
    total_mass    = 0.0
    frontier      = [(node_id, 0)]
    visited: Set[str] = {node_id}

    while frontier:
        n, hop = frontier.pop(0)
        contribution = DECAY ** hop
        if contribution < EPSILON_CUTOFF:
            continue
        for e in graph.edges_touching(n):
            if e.weight > 0:
                total_success += e.weight * contribution
            total_mass += contribution
            other = graph.other(e, n)
            if other not in visited:
                visited.add(other)
                frontier.append((other, hop + 1))

    if total_mass < MIN_MASS_FOR_LOCAL_STAT:
        return graph.EMA_path_success
    return total_success / total_mass


# ─────────────────────────────────────────────
# Section 3: Edge Update (Hebbian core)
# ─────────────────────────────────────────────

def update_edge(src: str, dst: str, reward: float, tick: int,
                graph: Graph) -> Optional[Edge]:
    if src == dst:
        return None   # no self-loops

    key = (src, dst)
    e = graph.edges.get(key)

    if e is None:
        for nid in (src, dst):
            if nid not in graph.nodes:
                graph.nodes[nid] = Node(
                    id=nid,
                    kind=NodeKind.ATOM,
                    created_at=tick,
                    last_active_at=tick,
                )
        e = Edge(
            src=src,
            dst=dst,
            weight=INITIAL_EDGE_WEIGHT,
            kind=EdgeKind.ORDINARY,
            created_at=tick,
            grace_until=tick + EXPLORATION_GRACE_TICKS,
        )
        graph.edges[key] = e
        graph._adj_add(src, dst)

    e.weight        += HEBBIAN_GAIN * reward
    graph.EMA_floor  = (1 - EMA_ALPHA) * graph.EMA_floor + EMA_ALPHA * e.weight
    e.weight         = max(e.weight, 0.0)
    e.access_count  += 1
    e.last_active_at = tick  # type: ignore[attr-defined]

    if reward > 0:
        e.last_reward_at = tick

    graph.EMA_edge = (1 - EMA_ALPHA) * graph.EMA_edge + EMA_ALPHA * e.weight

    return e


def protect_discount(protect: float) -> float:
    return protect


def _adaptive_decay_rate(graph: Graph) -> float:
    """
    Adaptive decay rate -- no fixed threshold.
    Computes current edge density (edges per node) and compares to
    the graph's own historical density EMA. When crowded above its
    own history, decay accelerates. When sparse, decay relaxes.
    The graph sets its own bar -- consistent with percentile philosophy.
    """
    n_nodes = max(len(graph.nodes), 1)
    current_density = len(graph.edges) / n_nodes
    # Update historical EMA
    graph.EMA_density = ((1 - EMA_ALPHA) * graph.EMA_density
                         + EMA_ALPHA * current_density)
    # Rate scales with how much denser we are than our own history
    # At 1x historical density: rate = DECAY_STEP (baseline)
    # At 2x historical density: rate = DECAY_STEP * 2
    # At 0.5x historical density: rate = DECAY_STEP * 0.5
    ratio = current_density / max(graph.EMA_density, 1.0)
    return DECAY_STEP * ratio


def decay_edge(e: Edge, tick: int, graph: Graph,
               adaptive_rate: float = DECAY_STEP) -> bool:
    """Apply decay. Returns True if edge was pruned."""
    if tick < e.grace_until:
        return False

    protect = PROTECT if (tick - e.last_reward_at) < PROTECT_WINDOW else 1.0
    e.weight -= adaptive_rate * (1 - protect_discount(protect))

    if e.weight <= NEAR_ZERO:
        key = e.key()
        del graph.edges[key]
        graph._adj_remove(key[0], key[1])
        return True
    return False


# ─────────────────────────────────────────────
# Section 4: Scoring / Soft Continuation
# ─────────────────────────────────────────────

def normalize(weight: float) -> float:
    return math.tanh(weight)


def score_candidate(candidate_edge: Edge, last_edge: Optional[Edge],
                    ema_ctx: dict) -> float:
    edge_act = normalize(candidate_edge.weight)
    novelty  = 1.0 / (1.0 + KAPPA * candidate_edge.access_count)

    if last_edge is not None:
        mild_relative = (1.0 if candidate_edge.weight >= last_edge.weight * 0.75
                         else 0.7 + 0.3 * (candidate_edge.weight /
                                            max(last_edge.weight, 1e-9)))
    else:
        mild_relative = 1.0

    abs_ok = (1.0 if candidate_edge.weight >= ema_ctx["local_edge"] * ABS_FLOOR_MULT
              else 0.4)

    length_pressure = max(0.0, (TARGET_LEN - ema_ctx["local_len"]) * LENGTH_GAIN)
    product = edge_act * novelty * mild_relative * abs_ok
    score   = product * (1 + length_pressure)
    return score


REJECT = object()   # sentinel


def step_continuation(current_node: str, last_edge: Optional[Edge],
                      tick: int, graph: Graph):
    """
    Section 4 -- returns next Edge or REJECT sentinel.

    Coarse-to-fine search priority:
      Level 1 (coarse) : HIGHWAY edges -- high-level region jump
      Level 2 (medium) : INDEX edges pointing to CHUNK nodes
      Level 3 (fine)   : ORDINARY atom-level edges

    Each level is only consulted if the previous level has no candidates.
    The traversal navigates at the highest available abstraction first,
    then zooms in -- never mixing all edge kinds together.
    """
    ema_ctx = {
        "local_edge": graph.EMA_edge if graph.EMA_edge > 0 else 0.1,
        "local_len":  graph.EMA_len,
    }

    all_outgoing = graph.outgoing_edges(current_node)
    if not all_outgoing:
        return REJECT

    # Level 1 (coarse): HIGHWAY edges
    highways = [e for e in all_outgoing if e.kind == EdgeKind.HIGHWAY]

    # Level 2 (medium): INDEX edges pointing to CHUNK nodes
    # INDEX edges are back-pointers from atoms/chunks to their parent chunk.
    # Following them navigates at chunk level.
    chunk_edges = [
        e for e in all_outgoing
        if e.kind == EdgeKind.INDEX
        and e.dst in graph.nodes
        and graph.nodes[e.dst].kind == NodeKind.CHUNK
    ]

    # Level 3 (fine): ORDINARY atom-level edges
    ordinary = [e for e in all_outgoing if e.kind == EdgeKind.ORDINARY]

    # Coarse-to-fine with fallthrough.
    # If the best edge at a given level leads to a dead end (no outgoing
    # edges from its destination), fall through to the next level.
    # This prevents chat dead-ends when highways/chunks are leaf nodes.
    def best_from(pool: list):
        if not pool:
            return None, -1.0
        scored = [(e, score_candidate(e, last_edge, ema_ctx)) for e in pool]
        return max(scored, key=lambda x: x[1])

    def has_onward_path(edge: Edge) -> bool:
        """True if the edge destination has any traversable outgoing edges."""
        dst_out = graph.outgoing_edges(edge.dst)
        return any(
            e.kind in (EdgeKind.ORDINARY, EdgeKind.HIGHWAY, EdgeKind.INDEX)
            for e in dst_out
        )

    # Try each level; fall through if best choice is a dead end
    best, best_score = None, -1.0
    for pool in (highways, chunk_edges, ordinary):
        if not pool:
            continue
        candidate, score = best_from(pool)
        if candidate is not None and has_onward_path(candidate):
            best, best_score = candidate, score
            break
        elif candidate is not None and best is None:
            # Keep as fallback even if dead end -- better than nothing
            best, best_score = candidate, score

    if best is None:
        return REJECT

    if best_score < SOFT_REJECT and best.weight < ema_ctx["local_edge"]:
        return REJECT

    graph.EMA_len = ((1 - EMA_ALPHA) * graph.EMA_len +
                     EMA_ALPHA * ema_ctx["local_len"])
    return best


# ─────────────────────────────────────────────
# Section 5: Chunk Promotion
# ─────────────────────────────────────────────

def promotion_threshold(graph: Graph,
                        percentile: float = CHUNK_PROMOTION_PERCENTILE) -> float:
    """
    Compute the Nth percentile of access_counts across all ordinary and
    highway edges. An edge must meet or exceed this threshold to be eligible
    for chunk promotion.

    Using percentile rather than a fixed constant means:
    - Early training: low bar, promotion fires readily on sparse graph
    - Late training: high bar, only genuinely dominant pairs promote
    - Self-regulating: the graph sets its own promotion bar
    """
    counts = [
        e.access_count
        for e in graph.edges.values()
        if e.kind in (EdgeKind.ORDINARY, EdgeKind.HIGHWAY)
    ]
    if not counts:
        return 1.0   # graph is empty

    counts.sort()
    idx = (percentile / 100.0) * (len(counts) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(counts) - 1)
    frac = idx - lo
    return counts[lo] * (1 - frac) + counts[hi] * frac


def chunk_constituents(node_id: str, graph: Graph) -> Set[str]:
    """
    Return the direct constituent members of a chunk node.
    Used to guard against circular promotion.
    """
    return {
        e.dst
        for e in graph.outgoing_edges(node_id)
        if e.kind == EdgeKind.CHUNK_CONSTITUENT
    }


def maybe_promote_chunk(src: str, dst: str, graph: Graph,
                        tick: int,
                        percentile: float = CHUNK_PROMOTION_PERCENTILE
                        ) -> Optional[Node]:
    """
    Attempt to promote the (src, dst) bigram into a CHUNK node.

    Works at any hierarchy level:
      atom  + atom  -> CHUNK  (level 1)
      chunk + chunk -> CHUNK  (level 2)
      ...and so on

    Two conditions must both pass:
      Abs condition: edge.access_count >= promotion_threshold(graph, percentile)
      Dom condition: edge.weight >= max_outgoing_weight * CHUNK_DOM_FRAC

    Guards:
      - No self-loops (src != dst)
      - No circular promotion: src cannot be a constituent of dst and vice versa
      - Edge must exist
      - Chunk not already created
    """
    if src == dst:
        return None

    key = (src, dst)
    if key not in graph.edges:
        return None

    edge = graph.edges[key]

    # Skip non-traversable edge kinds
    if edge.kind not in (EdgeKind.ORDINARY, EdgeKind.HIGHWAY):
        return None

    # ── Circular promotion guard ──────────────────────────────
    # Prevent a chunk from absorbing its own constituents into a higher chunk
    src_node = graph.nodes.get(src)
    dst_node = graph.nodes.get(dst)

    if src_node and src_node.kind == NodeKind.CHUNK:
        if dst in chunk_constituents(src, graph):
            return None   # dst is already a member of src

    if dst_node and dst_node.kind == NodeKind.CHUNK:
        if src in chunk_constituents(dst, graph):
            return None   # src is already a member of dst

    # -- Hierarchy level guard ---------------------------------------
    # A chunk can only promote with another node of the same kind.
    # atom+atom -> L1 chunk. L1chunk+L1chunk -> L2 chunk. etc.
    # This prevents newly created chunks from immediately cascading
    # into higher chunks before the atom layer has stabilised.
    src_kind = src_node.kind if src_node else NodeKind.ATOM
    dst_kind = dst_node.kind if dst_node else NodeKind.ATOM
    if src_kind != dst_kind:
        return None   # mixed levels cannot promote together

    # -- Abs condition: percentile threshold + dynamic floor ----------
    # Floor = 5% of edge count, minimum CHUNK_MIN_ACCESS.
    # More aggressive than before to prevent explosion.
    threshold     = promotion_threshold(graph, percentile)
    dynamic_floor = max(CHUNK_MIN_ACCESS,
                        len(graph.edges) // 20)  # 5% of edge count
    abs_condition = (edge.access_count >= threshold
                     and edge.access_count >= dynamic_floor)

    # ── Dom condition: edge dominates outgoing from src ───────
    outgoing = graph.outgoing_edges(src)
    max_out  = max((e.weight for e in outgoing), default=0.0)
    dom_condition = edge.weight >= max_out * CHUNK_DOM_FRAC

    if not (abs_condition and dom_condition):
        return None

    # ── Promotion ─────────────────────────────────────────────
    chunk_id = f"CHUNK_{src}_{dst}"
    if chunk_id in graph.nodes:
        return graph.nodes[chunk_id]   # already promoted

    chunk = Node(id=chunk_id, kind=NodeKind.CHUNK, created_at=tick)
    graph.nodes[chunk_id] = chunk

    # Constituent edges: chunk -> each member
    for member in (src, dst):
        ce = Edge(
            src=chunk_id, dst=member,
            weight=1.0,
            kind=EdgeKind.CHUNK_CONSTITUENT,
            created_at=tick,
            grace_until=tick + EXPLORATION_GRACE_TICKS,
        )
        graph.edges[(chunk_id, member)] = ce
        graph._adj_add(chunk_id, member)

    # Index edges: each member -> chunk (back-pointer, seeded weak)
    for member in (src, dst):
        ie = Edge(
            src=member, dst=chunk_id,
            weight=M3_SEED_WEIGHT,
            kind=EdgeKind.INDEX,
            created_at=tick,
            grace_until=tick + EXPLORATION_GRACE_TICKS,
        )
        graph.edges[(member, chunk_id)] = ie
        graph._adj_add(member, chunk_id)

    return chunk


# ─────────────────────────────────────────────
# Section 6: Highway Formation
# ─────────────────────────────────────────────

_path_success: Dict[Tuple[str, str], float] = {}


def _path_success_percentile(percentile: float = 90.0) -> float:
    """Nth percentile of accumulated path success values seen so far."""
    vals = sorted(_path_success.values())
    if not vals:
        return 0.0
    idx  = (percentile / 100.0) * (len(vals) - 1)
    lo   = int(idx)
    hi   = min(lo + 1, len(vals) - 1)
    return vals[lo] * (1 - (idx - lo)) + vals[hi] * (idx - lo)


def _graph_chunk_depth(graph: Graph) -> float:
    """
    Average hierarchy depth of chunk nodes.
    Atom = depth 0. L1 chunk (atom+atom) = depth 1.
    L2 chunk (chunk+chunk) = depth 2. Etc.
    Used to set a dynamic minimum path length for highway formation:
    highways should compress paths at least as long as one full chunk.
    """
    def depth(node_id: str, visited: set) -> int:
        if node_id in visited:
            return 0
        visited.add(node_id)
        node = graph.nodes.get(node_id)
        if node is None or node.kind == NodeKind.ATOM:
            return 0
        members = chunk_constituents(node_id, graph)
        if not members:
            return 1
        return 1 + max(depth(m, visited) for m in members)

    chunks = [n.id for n in graph.nodes.values()
              if n.kind == NodeKind.CHUNK]
    if not chunks:
        return 0.0
    depths = [depth(c, set()) for c in chunks]
    return sum(depths) / len(depths)


def _highway_capacity(graph: Graph) -> int:
    """Dynamic highway capacity: sqrt(node_count) * 2, minimum 10.
    The graph can maintain at most this many highways at once.
    New stronger highways displace weaker ones.
    """
    return max(10, int(math.sqrt(len(graph.nodes)) * 2))


def _enforce_highway_capacity(graph: Graph):
    """If highway count exceeds capacity, demote the weakest highways
    back to ORDINARY until we are within capacity.
    """
    capacity = _highway_capacity(graph)
    highways = [
        e for e in graph.edges.values()
        if e.kind == EdgeKind.HIGHWAY
    ]
    if len(highways) <= capacity:
        return
    # Sort weakest first, demote until within capacity
    highways.sort(key=lambda e: e.weight)
    for e in highways[:len(highways) - capacity]:
        e.kind = EdgeKind.ORDINARY


def on_path_complete(path: Path, tick: int, graph: Graph):
    """
    Dynamic highway formation with capacity constraint:
    1. Min path length = max(3, ceil(avg chunk depth * 2))
    2. Formation threshold = 90th percentile of all path success values
    3. Capacity = sqrt(nodes) * 2 -- new highway displaces weakest if full
    """
    # Dynamic minimum path length
    avg_depth  = _graph_chunk_depth(graph)
    min_length = max(3, math.ceil(avg_depth * 2))

    if len(path.nodes) < min_length or path.total_reward <= 0:
        return

    key = (path.nodes[0], path.nodes[-1])
    _path_success[key] = _path_success.get(key, 0.0) + path.path_activation

    # Dynamic threshold: 90th percentile of all path successes seen
    threshold = _path_success_percentile(90.0)

    if _path_success[key] >= threshold and threshold > 0:
        h = graph.edges.get(key)
        if h is None:
            h = update_edge(key[0], key[1], reward=path.total_reward,
                            tick=tick, graph=graph)
            if h is not None:
                h.kind = EdgeKind.HIGHWAY
        else:
            h.weight += FRACTION_OF_SUCCESS * _path_success[key]

        graph.EMA_path_success = ((1 - EMA_ALPHA) * graph.EMA_path_success +
                                   EMA_ALPHA * _path_success[key])

        # Enforce capacity: demote weakest highways if over limit
        _enforce_highway_capacity(graph)


# ─────────────────────────────────────────────
# Section 7: Exploration / Traversal Engine
# ─────────────────────────────────────────────

def explore_step(current_node: str, last_edge: Optional[Edge],
                 tick: int, epsilon: float, graph: Graph,
                 external_reward_fn, intrinsic_reward_fn):
    """Returns (next_node_id, reward) or (None, 0) if rejected."""
    if random.random() < epsilon:
        reachable = [n for n in graph.all_reachable_nodes(current_node)
                     if n != current_node]
        if not reachable:
            return None, 0.0
        candidate_id = random.choice(reachable)
        reward = external_reward_fn(candidate_id) + intrinsic_reward_fn(candidate_id)
        update_edge(current_node, candidate_id, reward, tick, graph)
        return candidate_id, reward
    else:
        result = step_continuation(current_node, last_edge, tick, graph)
        if result is REJECT:
            return None, 0.0
        reward = external_reward_fn(result.dst) + intrinsic_reward_fn(result.dst)
        update_edge(current_node, result.dst, reward, tick, graph)
        return result.dst, reward


def run_traversal(start_node: str, tick: int, epsilon: float,
                  graph: Graph, external_reward_fn,
                  intrinsic_reward_fn) -> Path:
    path      = Path(nodes=[start_node])
    node      = start_node
    last_edge: Optional[Edge] = None

    while len(path.nodes) < MAX_PATH_LEN:
        next_node, reward = explore_step(
            node, last_edge, tick, epsilon, graph,
            external_reward_fn, intrinsic_reward_fn
        )
        if next_node is None:
            break

        last_edge = graph.edges.get((node, next_node))
        path.nodes.append(next_node)
        path.total_reward    += reward
        path.path_activation += reward * (DECAY ** len(path.nodes))

        # ── Hierarchy promotion on every consecutive pair ─────
        # Works at any level: atom+atom, chunk+chunk, chunk+atom.
        # maybe_promote_chunk guards against circularity internally.
        maybe_promote_chunk(node, next_node, graph, tick)

        node = next_node

    on_path_complete(path, tick, graph)   # Section 6
    return path


# ─────────────────────────────────────────────
# Section 8: Structural Decay Pass
# ─────────────────────────────────────────────

def decay_pass(graph: Graph, tick: int):
    # Compute one adaptive rate for this entire pass
    rate = _adaptive_decay_rate(graph)
    for key in list(graph.edges.keys()):
        e = graph.edges.get(key)
        if e is not None:
            decay_edge(e, tick, graph, adaptive_rate=rate)

    orphaned = [nid for nid in list(graph.nodes)
                if graph.has_no_edges(nid)]
    for nid in orphaned:
        del graph.nodes[nid]


# ─────────────────────────────────────────────
# Section 9: Curriculum Scheduler
# ─────────────────────────────────────────────

@dataclass
class Stage:
    name: str
    target_density: float        # chunk_count / atom_count threshold
    target_highways: int         # highway edge count threshold
    suggested_epsilon: float     # exploration rate for this stage
    min_ticks: int = 400         # must spend at least this many ticks here
    domain: str = ""

    def readiness_check(self, stats: dict, ticks_in_stage: int) -> bool:
        if ticks_in_stage < self.min_ticks:
            return False
        atom_count  = max(stats.get("atoms", 1), 1)
        chunk_count = stats.get("chunks", 0)
        # chunks / (atoms + chunks): bounded 0-1, stable when chunks > atoms
        density     = chunk_count / (atom_count + chunk_count)
        highways    = stats.get("highways", 0)
        return density >= self.target_density and highways >= self.target_highways


class CurriculumScheduler:
    """Read-only observer -- zero write access to graph weights."""

    def __init__(self, stages: List[Stage]):
        self.stages         = stages
        self.current        = 0
        self.ticks_in_stage = 0

    @property
    def current_stage(self) -> Stage:
        return self.stages[self.current]

    def is_complete(self) -> bool:
        return self.current >= len(self.stages)

    def current_exploration_rate(self) -> float:
        if self.is_complete():
            return 0.05
        return self.current_stage.suggested_epsilon

    def step(self, graph: Graph):
        """Advance stage if ready. Returns current stage or None if complete."""
        if self.is_complete():
            return None

        self.ticks_in_stage += 1
        stats = graph.stats()
        stage = self.current_stage

        if stage.readiness_check(stats, self.ticks_in_stage):
            self.current        += 1
            self.ticks_in_stage  = 0
            if not self.is_complete():
                print(f"  [curriculum] Advanced to stage: "
                      f"{self.current_stage.name} "
                      f"(density={stats['chunks']}/{max(stats['atoms'],1):.0f}, "
                      f"highways={stats['highways']})")

        return self.current_stage if not self.is_complete() else None

    def feed_tokens(self, tokens: List[str], tick: int, graph: Graph):
        """
        Creates/updates atom nodes + ordinary edges from a token sequence.
        Also attempts chunk promotion on consecutive atom pairs.
        Traversal-level promotion (including chunk+chunk) is handled
        inside run_traversal.
        """
        for i, tok in enumerate(tokens):
            if tok not in graph.nodes:
                graph.nodes[tok] = Node(id=tok, kind=NodeKind.ATOM,
                                        created_at=tick)
            if i > 0:
                update_edge(tokens[i - 1], tok, reward=0.1,
                            tick=tick, graph=graph)

        # Promotion happens only inside run_traversal (on rewarded paths).
        # feed_tokens only creates atoms and ordinary edges.