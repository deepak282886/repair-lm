"""
finetune.py
===========
Finetuning: load QA / instruction pairs and build relational structure
on top of the pretrained Re-Pair grammar.

What finetuning adds over pretraining
--------------------------------------
  - Populates long-term memory with explicit labelled Q/A facts
  - Grows the fact corpus so relation clustering has evidence to work with
  - Runs data-driven relation clustering (when corpus >= 50 facts)
  - Applies a reward pass: measures how often each pipeline stage resolves
    correctly so you can see where the model is strong or weak

The relation clustering step is what connects the trees the grammar built:
templates that produce overlapping answer values get merged into the same
cluster, and the predicate gate uses those clusters to suppress false positives
during soft matching.

Usage
-----
    python finetune.py --load pretrained.pkl --save finetuned.pkl
    python finetune.py --load pretrained.pkl --dataset tatsu-lab/alpaca
    python finetune.py --load pretrained.pkl --pairs 5000 --no-cluster
"""

import sys
import argparse


# ── data loading ──────────────────────────────────────────────────────────────

def load_qa_pairs(n: int, dataset: str) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets")
        sys.exit(1)

    print(f"Loading up to {n:,} QA pairs from {dataset} …")
    try:
        ds = load_dataset(dataset, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    pairs: list[tuple[str, str]] = []
    for ex in ds:
        # Support multiple dataset formats
        q = (ex.get("instruction") or ex.get("question") or
             ex.get("input") or "").strip()
        a = (ex.get("output") or ex.get("answer") or
             ex.get("response") or "").strip()
        if q and a and len(q.split()) <= 60 and len(a.split()) <= 100:
            pairs.append((q, a))
        if len(pairs) >= n:
            break

    print(f"Loaded {len(pairs):,} QA pairs.")
    return pairs


# ── reward evaluation ─────────────────────────────────────────────────────────

def reward_pass(model, pairs: list[tuple[str, str]],
                sample_size: int = 200) -> None:
    """
    Run a sample of pairs through the full pipeline.
    Print resolution source distribution and exact-match accuracy.
    This is the reward signal: it shows which pipeline stages are working
    and where to focus data collection or tuning effort.
    """
    sample = pairs[:sample_size]
    if not sample:
        return

    print(f"\nReward pass on {len(sample)} pairs …")
    stats = model.evaluate(sample)

    total   = stats.pop("total", len(sample))
    correct = stats.pop("correct", 0)
    print(f"  Resolution sources:")
    for source, count in stats.items():
        pct = 100 * count / total if total else 0
        print(f"    {source:<14} {count:>4}  ({pct:>5.1f}%)")
    print(f"  Exact-match accuracy: {correct}/{total} "
          f"({100*correct/total:.1f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Re-Pair LM — Finetuning")
    p.add_argument("--load",       type=str, default="pretrained.pkl",
                   help="pretrained model to load (default: pretrained.pkl)")
    p.add_argument("--save",       type=str, default="finetuned.pkl",
                   help="output path for finetuned model (default: finetuned.pkl)")
    p.add_argument("--pairs",      type=int, default=2_000,
                   help="QA pairs to load for finetuning (default 2000)")
    p.add_argument("--dataset",    type=str, default="tatsu-lab/alpaca",
                   help="HuggingFace instruction dataset (default: tatsu-lab/alpaca)")
    p.add_argument("--no-cluster", action="store_true",
                   help="skip relation clustering step")
    p.add_argument("--force-cluster", action="store_true",
                   help="run clustering even if below 50-fact threshold")
    args = p.parse_args()

    from repair_lm.model import RePairLM

    model = RePairLM.load(args.load)
    pairs = load_qa_pairs(args.pairs, args.dataset)

    # Store all pairs as facts
    print(f"\nStoring {len(pairs):,} facts in memory …")
    stored = 0
    for q, a in pairs:
        model.learn_fact(q, a)
        stored += 1
    print(f"Memory now contains {len(model.memory):,} facts.")

    # Relation clustering
    if not args.no_cluster:
        print("\nRunning relation clustering …")
        ran = model.refit_clustering(force=args.force_cluster)
        if ran:
            model.clustering.print_clusters()
    else:
        print("Skipping relation clustering (--no-cluster).")

    # Reward pass
    reward_pass(model, pairs)

    model.save(args.save)
    print(f"\nFinetuning complete. Model saved to {args.save}")


if __name__ == "__main__":
    main()
