"""Compare fixed next-token prompts across two TyPoTaKer checkpoints."""

from __future__ import annotations

import argparse

from typotaker.infer import load_run, next_words


PROMPTS = [
    "The server listens",
    "The user should be able",
    "Please check the configuration",
    "I would like to",
    "When the model finishes",
    "The most important thing is",
    "If you want to improve",
    "Where you go I go",
    "Fahim is a",
    "Grammar correction should",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", default="checkpoints/lm_stage0")
    parser.add_argument("--stage1", default="checkpoints/lm_stage1")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for label, run in (("stage0", args.stage0), ("stage1", args.stage1)):
        model, tok = load_run(run, args.tokenizer, device=args.device)
        print(f"MODEL {label} RUN {run}")
        for prompt in PROMPTS:
            print(f"PROMPT {prompt!r}")
            for piece, probability in next_words(model, tok, prompt, k=args.k):
                print(f"  {probability:.4f}\t{piece!r}")
        print()


if __name__ == "__main__":
    main()
