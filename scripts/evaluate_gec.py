from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import torch

from typotaker.infer import correct_sentence, load_run


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    target: str
    clean: bool = False
    category: str = "smoke"


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


def load_cases(path: str | None) -> list[Case]:
    if path is None:
        return CASES
    cases: list[Case] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                cases.append(
                    Case(
                        name=str(obj["name"]),
                        source=str(obj["source"]),
                        target=str(obj["target"]),
                        clean=bool(obj.get("clean", False)),
                        category=str(obj.get("category", "uncategorized")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"missing {exc} at {path}:{line_number}") from exc
    if not cases:
        raise ValueError(f"no cases found in {path}")
    return cases


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
    cases: list[Case],
    show_limit: int,
) -> dict:
    model, tok = load_run(run, tokenizer, device=device)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # One warm-up avoids counting the first CUDA/kernel initialization as normal latency.
    correct_sentence(model, tok, cases[0].source, maps, min_edit_prob, edit_margin)
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        output = correct_sentence(model, tok, case.source, maps, min_edit_prob, edit_margin)
        latency_ms = (time.perf_counter() - t0) * 1000
        rows.append((case, output, latency_ms))

    error_rows = [(c, o) for c, o, _ in rows if not c.clean]
    typo_rows = [(c, o) for c, o in error_rows if c.category == "typo"]
    clean_rows = [(c, o) for c, o, _ in rows if c.clean]
    per_case_ops = [
        (edit_ops(case.source, case.target), edit_ops(case.source, output))
        for case, output in error_rows
    ]
    precision, recall, f05 = prf05(per_case_ops)

    exact = sum(norm(o) == norm(c.target) for c, o, _ in rows)
    error_exact = sum(norm(o) == norm(c.target) for c, o in error_rows)
    typo_exact = sum(norm(o) == norm(c.target) for c, o in typo_rows)
    clean_unchanged = sum(norm(o) == norm(c.source) for c, o in clean_rows)
    clean_false_edits = sum(norm(o) != norm(c.source) for c, o in clean_rows)
    changed_error = sum(norm(o) != norm(c.source) for c, o in error_rows)
    mean_latency = sum(lat for _, _, lat in rows) / len(rows)
    peak_mb = 0.0
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"\nMODEL {run}")
    shown_rows = rows if show_limit <= 0 else rows[:show_limit]
    for case, output, latency in shown_rows:
        print(f"[{case.name}] category={case.category} clean={case.clean} latency_ms={latency:.2f}")
        print(f"  source:  {case.source}")
        print(f"  target:  {case.target}")
        print(f"  output:  {output}")
    if len(shown_rows) < len(rows):
        print(f"  ... {len(rows) - len(shown_rows)} additional cases omitted; use --show-limit 0 to print all")
    print("SUMMARY")
    print(f"  exact_match_all={exact}/{len(rows)} ({exact / len(rows):.3f})")
    print(f"  error_recovery={error_exact}/{len(error_rows)} ({error_exact / len(error_rows):.3f})")
    if typo_rows:
        print(f"  typo_recovery={typo_exact}/{len(typo_rows)} ({typo_exact / len(typo_rows):.3f})")
    print(f"  clean_unchanged={clean_unchanged}/{len(clean_rows)} ({clean_unchanged / len(clean_rows):.3f})")
    print(f"  clean_false_edits={clean_false_edits}/{len(clean_rows)} ({clean_false_edits / len(clean_rows):.3f})")
    print(f"  error_changed={changed_error}/{len(error_rows)} ({changed_error / len(error_rows):.3f})")
    print(f"  edit_precision={precision:.3f}")
    print(f"  edit_recall={recall:.3f}")
    print(f"  edit_f0.5={f05:.3f}")
    print(f"  mean_latency_ms={mean_latency:.2f}")
    print(f"  peak_cuda_memory_mb={peak_mb:.1f}")
    category_rows: dict[str, list[tuple[Case, str]]] = {}
    for case, output, _ in rows:
        category_rows.setdefault(case.category, []).append((case, output))
    print("CATEGORY_SUMMARY")
    for category, category_values in sorted(category_rows.items()):
        total = len(category_values)
        exact_category = sum(norm(output) == norm(case.target) for case, output in category_values)
        print(f"  {category}: exact={exact_category}/{total} ({exact_category / total:.3f})")

    return {
        "exact": exact / len(rows),
        "error_recovery": error_exact / len(error_rows),
        "typo_recovery": typo_exact / len(typo_rows) if typo_rows else 0.0,
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
    parser.add_argument("--cases", help="JSONL held-out cases; defaults to the 16-case smoke suite")
    parser.add_argument("--show-limit", type=int, default=24, help="rows shown per model; 0 shows all")
    args = parser.parse_args()
    device = None if args.device == "auto" else args.device
    cases = load_cases(args.cases)
    print(f"EVALUATION_CASES count={len(cases)} source={args.cases or 'built-in-smoke'}")

    results = {
        "stage0": evaluate_run(
            args.stage0,
            args.maps,
            args.tokenizer,
            device,
            args.min_edit_prob,
            args.edit_margin,
            cases,
            args.show_limit,
        ),
        "stage1": evaluate_run(
            args.stage1,
            args.maps,
            args.tokenizer,
            device,
            args.min_edit_prob,
            args.edit_margin,
            cases,
            args.show_limit,
        ),
    }
    print("\nCOMPARISON")
    for name, values in results.items():
        print(
            f"{name}: error_recovery={values['error_recovery']:.3f} "
            f"typo_recovery={values['typo_recovery']:.3f} "
            f"clean_unchanged={values['clean_unchanged']:.3f} "
            f"precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} "
            f"F0.5={values['f05']:.3f}"
        )


if __name__ == "__main__":
    main()
