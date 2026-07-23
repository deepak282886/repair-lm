"""
generate.py
===========
Generation and inference from a trained Re-Pair LM.

Two modes
---------
  qa       : run the full resolution pipeline (memory → composition →
             soft match → discovery → grammar)
  generate : grammar-only open-ended generation

Usage
-----
    python generate.py --load finetuned.pkl
    python generate.py --load finetuned.pkl --prompt "Who discovered penicillin"
    python generate.py --load finetuned.pkl --mode generate --batch 10
    python generate.py --load finetuned.pkl --mode both --max-symbols 60
"""

import argparse


def print_result(result: dict, mode: str) -> None:
    if mode == "qa":
        print(f"  [{result['source']} | score={result['score']:.3f}]  "
              f"{result['answer']}")
    else:
        print(f"  {result['answer']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Re-Pair LM — Generation")
    p.add_argument("--load",        type=str,  default="finetuned.pkl",
                   help="model to load (default: finetuned.pkl)")
    p.add_argument("--prompt",      type=str,  default=None,
                   help="single prompt to resolve/generate from")
    p.add_argument("--batch",       type=int,  default=5,
                   help="unconditional samples to generate if no prompt (default 5)")
    p.add_argument("--max-symbols", type=int,  default=40,
                   help="max grammar symbols per generation step (default 40)")
    p.add_argument("--mode",
                   choices=["qa", "generate", "both"],
                   default="both",
                   help="qa=pipeline only, generate=grammar only, both=both")
    p.add_argument("--no-interactive", action="store_true",
                   help="skip interactive prompt loop after batch output")
    args = p.parse_args()

    from repair_lm.model import RePairLM
    model = RePairLM.load(args.load)

    # ── single prompt ─────────────────────────────────────────────────────────
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        if args.mode in ("qa", "both"):
            result = model.answer(args.prompt)
            print_result(result, "qa")
        if args.mode in ("generate", "both"):
            gen = model.generate(prompt=args.prompt, max_symbols=args.max_symbols)
            print(f"  [grammar]  {gen}")

    # ── unconditional batch ───────────────────────────────────────────────────
    else:
        print(f"\n── {args.batch} unconditional samples ──")
        for i in range(args.batch):
            gen = model.generate(max_symbols=args.max_symbols)
            print(f"[{i+1}] {gen}")

    # ── interactive loop ──────────────────────────────────────────────────────
    if not args.no_interactive:
        print("\n── Interactive mode (blank line to quit) ──")
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt:
                break

            if args.mode in ("qa", "both"):
                result = model.answer(prompt)
                print_result(result, "qa")

            if args.mode in ("generate", "both"):
                gen = model.generate(prompt=prompt, max_symbols=args.max_symbols)
                print(f"  [grammar]  {gen}")

    print("\nDone.")


if __name__ == "__main__":
    main()
