from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

import torch

from typotaker.infer import correct_sentence, load_run


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    target: str
    clean: bool = False


# This is a small, fixed smoke-quality gate, not a substitute for a held-out
# corpus. Keeping it in code makes Stage 0/Stage 1 comparisons reproducible.
CASES = [
    Case("subject-verb-1", "I has a book.", "I have a book."),
    Case("subject-verb-2", "She go to the store.", "She goes to the store."),
    Case("subject-verb-3", "They is working today.", "They are working today."),
    Case("subject-verb-4", "We was ready for the test.", "We were ready for the test."),
    Case("article-typo", "This is teh server.", "This is the server."),
    Case("verb-form", "The server listen on port 8000.", "The server listens on port 8000."),
    Case("word-typo", "Please send the file tomorow.", "Please send the file tomorrow."),
    Case("redundant-word", "I am agree with you.", "I agree with you."),
    Case("possessive-verb", "She have two cats.", "She has two cats."),
    Case("contraction", "He dont know the answer.", "He does not know the answer."),
    Case("clean-server", "The server listens on port 8000.", "The server listens on port 8000.", True),
    Case("clean-question", "Can you check the configuration?", "Can you check the configuration?", True),
    Case("clean-ready", "I know you are ready.", "I know you are ready.", True),
    Case("clean-report", "Please send the report today.", "Please send the report today.", True),
    Case("clean-project", "They are working on the project.", "They are working on the project.", True),
    Case("clean-same", "This is the same thing.", "This is the same thing.", True),
]


def norm(text: str) -> str:
    return " ".join(text.split()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", norm(text), flags=re.UNICODE)


def edit_ops(source: str, other: str) -> set[tuple[str, int, int, tuple[str, ...], tuple[str, ...]]]:
    src = words(source)
    dst = words(other)
    ops = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=src, b=dst, autojunk=False).get_opcodes():
        if tag != "equal":
            ops.add((tag, i1, i2, tuple(src[i1:i2]), tuple(dst[j1:j2])))
    return ops


def prf05(per_case: list[tuple[set, set]]) -> tuple[float, float, float]:
    # Score each sentence independently. Otherwise the same edit position in
    # unrelated sentences can be incorrectly counted as a true positive.
    tp = sum(len(gold & predicted) for gold, predicted in per_case)
    gold_count = sum(len(gold) for gold, _ in per_case)
    predicted_count = sum(len(predicted) for _, predicted in per_case)
    precision = tp / predicted_count if predicted_count else (1.0 if not gold_count else 0.0)
    recall = tp / gold_count if gold_count else (1.0 if not predicted_count else 0.0)
    beta2 = 0.25
    f05 = ((1 + beta2) * precision * recall / (beta2 * precision + recall)) if precision + recall else 0.0
    return precision, recall, f05


def evaluate_run(
    run: str,
    maps: str,
    tokenizer: str,
    device: str | None,
    min_edit_prob: float,
    edit_margin: float,
) -> dict:
    model, tok = load_run(run, tokenizer, device=device)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # One warm-up avoids counting the first CUDA/kernel initialization as normal latency.
    correct_sentence(model, tok, CASES[0].source, maps, min_edit_prob, edit_margin)
    rows = []
    for case in CASES:
        t0 = time.perf_counter()
        output = correct_sentence(model, tok, case.source, maps, min_edit_prob, edit_margin)
        latency_ms = (time.perf_counter() - t0) * 1000
        rows.append((case, output, latency_ms))

    typo_rows = [(c, o) for c, o, _ in rows if not c.clean]
    clean_rows = [(c, o) for c, o, _ in rows if c.clean]
    per_case_ops = [
        (edit_ops(case.source, case.target), edit_ops(case.source, output))
        for case, output in typo_rows
    ]
    precision, recall, f05 = prf05(per_case_ops)

    exact = sum(norm(o) == norm(c.target) for c, o, _ in rows)
    typo_exact = sum(norm(o) == norm(c.target) for c, o in typo_rows)
    clean_unchanged = sum(norm(o) == norm(c.source) for c, o in clean_rows)
    clean_false_edits = sum(norm(o) != norm(c.source) for c, o in clean_rows)
    changed_typo = sum(norm(o) != norm(c.source) for c, o in typo_rows)
    mean_latency = sum(lat for _, _, lat in rows) / len(rows)
    peak_mb = 0.0
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"\nMODEL {run}")
    for case, output, latency in rows:
        print(f"[{case.name}] clean={case.clean} latency_ms={latency:.2f}")
        print(f"  source:  {case.source}")
        print(f"  target:  {case.target}")
        print(f"  output:  {output}")
    print("SUMMARY")
    print(f"  exact_match_all={exact}/{len(rows)} ({exact / len(rows):.3f})")
    print(f"  typo_recovery={typo_exact}/{len(typo_rows)} ({typo_exact / len(typo_rows):.3f})")
    print(f"  clean_unchanged={clean_unchanged}/{len(clean_rows)} ({clean_unchanged / len(clean_rows):.3f})")
    print(f"  clean_false_edits={clean_false_edits}/{len(clean_rows)} ({clean_false_edits / len(clean_rows):.3f})")
    print(f"  typo_changed={changed_typo}/{len(typo_rows)} ({changed_typo / len(typo_rows):.3f})")
    print(f"  edit_precision={precision:.3f}")
    print(f"  edit_recall={recall:.3f}")
    print(f"  edit_f0.5={f05:.3f}")
    print(f"  mean_latency_ms={mean_latency:.2f}")
    print(f"  peak_cuda_memory_mb={peak_mb:.1f}")
    return {
        "exact": exact / len(rows),
        "typo_recovery": typo_exact / len(typo_rows),
        "clean_unchanged": clean_unchanged / len(clean_rows),
        "clean_false_edits": clean_false_edits / len(clean_rows),
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "latency_ms": mean_latency,
        "peak_cuda_memory_mb": peak_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed GEC cases across checkpoints")
    parser.add_argument("--stage0", default="checkpoints/gec_stage0")
    parser.add_argument("--stage1", default="checkpoints/gec_stage1")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--maps", default="data/processed/gec/edit_maps.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-edit-prob", type=float, default=0.0)
    parser.add_argument("--edit-margin", type=float, default=0.0)
    args = parser.parse_args()
    device = None if args.device == "auto" else args.device

    results = {
        "stage0": evaluate_run(
            args.stage0,
            args.maps,
            args.tokenizer,
            device,
            args.min_edit_prob,
            args.edit_margin,
        ),
        "stage1": evaluate_run(
            args.stage1,
            args.maps,
            args.tokenizer,
            device,
            args.min_edit_prob,
            args.edit_margin,
        ),
    }
    print("\nCOMPARISON")
    for name, values in results.items():
        print(
            f"{name}: typo_recovery={values['typo_recovery']:.3f} "
            f"clean_unchanged={values['clean_unchanged']:.3f} "
            f"precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} "
            f"F0.5={values['f05']:.3f}"
        )


if __name__ == "__main__":
    main()
