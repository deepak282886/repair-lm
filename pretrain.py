"""
pretrain.py
===========
Pretraining: run joint Re-Pair compression over a HuggingFace text corpus.

Builds the grammar that forms the generative backbone of the Re-Pair LM.
More data → more rule reuse → richer non-terminals → better generation.

Each sentence pair (sentence[i], sentence[i+1]) is stored in long-term memory
so the model learns sentence-level continuations as well as grammar structure.

Usage
-----
    python pretrain.py
    python pretrain.py --sentences 50000 --rules 2000 --save pretrained.pkl
    python pretrain.py --dataset allenai/dolma --config default --sentences 20000
"""

import re
import sys
import argparse


# ── data loading ──────────────────────────────────────────────────────────────

def load_sentences(n: int, dataset: str, config: str) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets")
        sys.exit(1)

    print(f"Streaming {n:,} sentences from {dataset} (config={config}) …")
    ds = load_dataset(dataset, name=config, split="train",
                      streaming=True, trust_remote_code=True)

    sentences: list[str] = []
    for example in ds:
        text = example.get("text", "")
        # Split on sentence-ending punctuation (heuristic)
        for s in re.split(r"(?<=[.!?])\s+", text.strip()):
            s = s.strip()
            word_count = len(s.split())
            if 8 <= word_count <= 60:   # skip stubs and walls of text
                sentences.append(s)
            if len(sentences) >= n:
                print(f"Loaded {len(sentences):,} sentences.")
                return sentences

    print(f"Loaded {len(sentences):,} sentences (corpus exhausted).")
    return sentences


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Re-Pair LM — Pretraining")
    p.add_argument("--sentences", type=int, default=5_000,
                   help="sentences to stream for pretraining (default 5000)")
    p.add_argument("--rules",     type=int, default=500,
                   help="max grammar rules to learn (default 500)")
    p.add_argument("--min-freq",  type=int, default=2,
                   help="stop when best pair frequency drops below this (default 2)")
    p.add_argument("--dataset",   type=str, default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset name (default: FineWeb-Edu)")
    p.add_argument("--config",    type=str, default="sample-10BT",
                   help="HuggingFace dataset config (default: sample-10BT)")
    p.add_argument("--save",      type=str, default="pretrained.pkl",
                   help="output path for saved model (default: pretrained.pkl)")
    args = p.parse_args()

    from repair_lm.model import RePairLM

    model     = RePairLM()
    sentences = load_sentences(args.sentences, args.dataset, args.config)

    # Store consecutive sentence pairs in long-term memory.
    # This teaches the model sentence-level continuation structure
    # on top of the token-level grammar structure.
    print(f"Storing {len(sentences) - 1:,} sentence-continuation facts …")
    for i in range(len(sentences) - 1):
        model.learn_fact(sentences[i], sentences[i + 1])

    # Run joint Re-Pair over the full corpus
    print("\nRunning joint Re-Pair compression …")
    model.grammar.train(sentences,
                        max_rules=args.rules,
                        min_freq=args.min_freq)
    model.grammar.stats()
    model.grammar.top_rules(10)

    model.save(args.save)
    print(f"\nPretraining complete. Model saved to {args.save}")


if __name__ == "__main__":
    main()
