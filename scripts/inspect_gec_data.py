"""Analyze token-level edit label distributions in a GEC jsonl file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_gec_file(path: str, max_rows: int | None = None) -> dict[str, object]:
    counts: Counter[int] = Counter()
    total_rows = 0
    total_tokens = 0
    clean_identical_rows = 0
    row_lengths = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            labels = obj["edit_labels"]
            counts.update(labels)
            total_rows += 1
            total_tokens += len(labels)
            row_lengths.append(len(labels))
            if obj.get("src") == obj.get("clean"):
                clean_identical_rows += 1
            if max_rows and total_rows >= max_rows:
                break

    keep_count = counts.get(0, 0)
    delete_count = counts.get(1, 0)
    case_cap_count = counts.get(2, 0)
    case_lower_count = counts.get(3, 0)
    replace_count = sum(v for k, v in counts.items() if 4 <= k < 2052)
    append_count = sum(v for k, v in counts.items() if k >= 2052)
    non_keep_count = total_tokens - keep_count

    row_lengths.sort()
    median_len = row_lengths[len(row_lengths) // 2] if row_lengths else 0
    mean_len = (total_tokens / total_rows) if total_rows else 0

    return {
        "path": path,
        "total_rows": total_rows,
        "clean_identical_rows": clean_identical_rows,
        "clean_identical_pct": (clean_identical_rows / total_rows * 100) if total_rows else 0,
        "total_tokens": total_tokens,
        "mean_tokens_per_row": mean_len,
        "median_tokens_per_row": median_len,
        "keep_count": keep_count,
        "keep_pct": (keep_count / total_tokens * 100) if total_tokens else 0,
        "non_keep_count": non_keep_count,
        "non_keep_pct": (non_keep_count / total_tokens * 100) if total_tokens else 0,
        "delete_count": delete_count,
        "delete_pct": (delete_count / total_tokens * 100) if total_tokens else 0,
        "case_cap_count": case_cap_count,
        "case_lower_count": case_lower_count,
        "replace_count": replace_count,
        "replace_pct": (replace_count / total_tokens * 100) if total_tokens else 0,
        "append_count": append_count,
        "append_pct": (append_count / total_tokens * 100) if total_tokens else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GEC dataset label distributions")
    parser.add_argument("paths", nargs="+", help="Paths to GEC train.jsonl files")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to sample")
    args = parser.parse_args()

    for path in args.paths:
        res = analyze_gec_file(path, args.max_rows)
        print(f"\n--- DATASET: {res['path']} ---")
        print(f"Total Rows:              {res['total_rows']:,}")
        print(f"Clean Identity Rows:     {res['clean_identical_rows']:,} ({res['clean_identical_pct']:.2f}%)")
        print(f"Total Tokens:            {res['total_tokens']:,}")
        print(f"Mean Tokens/Row:         {res['mean_tokens_per_row']:.1f} (Median: {res['median_tokens_per_row']})")
        print(f"KEEP Labels (0):         {res['keep_count']:,} ({res['keep_pct']:.2f}%)")
        print(f"NON-KEEP Labels:         {res['non_keep_count']:,} ({res['non_keep_pct']:.2f}%)")
        print(f"  DELETE (1):            {res['delete_count']:,} ({res['delete_pct']:.2f}%)")
        print(f"  CASE_CAP (2):          {res['case_cap_count']:,}")
        print(f"  CASE_LOWER (3):        {res['case_lower_count']:,}")
        print(f"  REPLACE (4..2051):     {res['replace_count']:,} ({res['replace_pct']:.2f}%)")
        print(f"  APPEND (>=2052):       {res['append_count']:,} ({res['append_pct']:.2f}%)")


if __name__ == "__main__":
    main()
