"""
Generator — turns graph traversal into language.

After training, call talk(graph, prompt) to get a response.
The graph IS the language model. No neural network.
"""

import re
import random
from typing import List, Optional
from core import (
    Graph, Node, NodeKind, EdgeKind,
    step_continuation, REJECT, run_traversal,
)


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s']", " ", text)
    return [t for t in text.split() if t]


# ─────────────────────────────────────────────
# Entry point finder
# ─────────────────────────────────────────────

def find_entry_nodes(tokens: List[str], graph: Graph) -> List[str]:
    """Return tokens that exist as nodes in the graph."""
    return [t for t in tokens if t in graph.nodes]


# ─────────────────────────────────────────────
# Generation — pure exploitation (epsilon=0)
# ─────────────────────────────────────────────

def generate(start_tokens: List[str],
             graph: Graph,
             max_tokens: int = 20,
             temperature: float = 0.0) -> List[str]:
    """
    Generate a token sequence by traversing the graph.

    temperature=0.0  : always pick highest-scoring edge (greedy)
    temperature>0.0  : sample from top edges with some randomness
    """
    if not start_tokens:
        return []

    # Find best entry point — node with most outgoing edges
    entry_candidates = find_entry_nodes(start_tokens, graph)
    if not entry_candidates:
        return ["[unknown]"]

    # Pick entry with highest out-degree
    entry = max(
        entry_candidates,
        key=lambda n: len(graph.outgoing_edges(n))
    )

    output = list(start_tokens)   # echo input tokens
    current = entry
    last_edge = None

    for _ in range(max_tokens):
        outgoing = graph.outgoing_edges(current)
        if not outgoing:
            break

        # Filter out chunk/index edges — only follow ordinary + highway
        candidates = [
            e for e in outgoing
            if e.kind in (EdgeKind.ORDINARY, EdgeKind.HIGHWAY)
            and e.dst not in output[-3:]   # avoid recent repeats
        ]
        if not candidates:
            candidates = outgoing

        if temperature == 0.0:
            # Greedy — pick highest weight
            best = max(candidates, key=lambda e: e.weight)
        else:
            # Sample proportional to weight^(1/temperature)
            weights = [max(e.weight, 1e-9) ** (1.0 / temperature)
                       for e in candidates]
            total = sum(weights)
            r = random.random() * total
            cumulative = 0.0
            best = candidates[-1]
            for e, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    best = e
                    break

        next_tok = best.dst
        # Skip chunk nodes in output
        if graph.nodes.get(next_tok, None) and \
           graph.nodes[next_tok].kind == NodeKind.CHUNK:
            current = next_tok
            last_edge = best
            continue

        output.append(next_tok)
        last_edge = best
        current = next_tok

    # Remove echoed input from output
    response = output[len(start_tokens):]
    return response


# ─────────────────────────────────────────────
# Chat interface
# ─────────────────────────────────────────────

def talk(graph: Graph,
         prompt: str,
         max_tokens: int = 20,
         temperature: float = 0.3) -> str:
    """
    High-level: text in → text out.
    """
    tokens = tokenize(prompt)
    if not tokens:
        return "(empty input)"

    entry_nodes = find_entry_nodes(tokens, graph)
    if not entry_nodes:
        return f"(none of {tokens} are in the graph yet — keep training)"

    response_tokens = generate(
        start_tokens  = tokens,
        graph         = graph,
        max_tokens    = max_tokens,
        temperature   = temperature,
    )

    if not response_tokens:
        return "(graph reached dead end — keep training)"

    return " ".join(response_tokens)


def graph_stats_summary(graph: Graph) -> str:
    stats = graph.stats()
    return (
        f"nodes={stats['nodes']} | "
        f"atoms={stats['atoms']} | "
        f"chunks={stats['chunks']} | "
        f"highways={stats['highways']} | "
        f"edges={stats['edges']}"
    )


# ─────────────────────────────────────────────
# Interactive chat loop
# ─────────────────────────────────────────────

def chat_loop(graph: Graph, temperature: float = 0.3):
    print("\n" + "="*55)
    print("  Talking to the graph")
    print("  Type a word or phrase. Type 'quit' to exit.")
    print("  Type 'stats' for graph info.")
    print("  Type 'temp 0.5' to adjust temperature.")
    print("="*55 + "\n")

    t = temperature
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(f"  {graph_stats_summary(graph)}\n")
            continue
        if user_input.lower().startswith("temp "):
            try:
                t = float(user_input.split()[1])
                print(f"  temperature set to {t}\n")
            except Exception:
                print("  usage: temp 0.5\n")
            continue

        response = talk(graph, user_input, max_tokens=20, temperature=t)
        print(f"Graph: {response}\n")