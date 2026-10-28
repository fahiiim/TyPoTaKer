"""
Bridge script: Converts Grammarly/QuillBot CSV dataset into token-aligned JSONL
and edit vocabulary maps directly ingestible by TypoFormer's training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from typotaker.data.edits import align_edits, build_freq_maps
from typotaker.data.jsonl import write_jsonl
from typotaker.tokenizer import TypoTokenizer


def convert_csv_to_gec(
    csv_path: str | Path,
    tokenizer_path: str | Path,
    out_jsonl: str | Path,
    maps_path: str | Path | None = None,
    base_maps_path: str | Path | None = None,
    k_replace: int = 2048,
    k_append: int = 2048,
    max_chars: int = 16,
) -> None:
    csv_path = Path(csv_path)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if maps_path is None:
        maps_path = out_jsonl.with_name("edit_maps.json")
    else:
        maps_path = Path(maps_path)

    tok = TypoTokenizer(str(tokenizer_path), max_chars=max_chars)

    print(f"Reading CSV dataset from {csv_path}...")
    pairs: list[tuple[str, str, dict[str, str]]] = []
    all_tgt_ids: list[int] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["input_text"].strip()
            tgt = row["target_text"].strip()
            if not src or not tgt:
                continue
            pairs.append((src, tgt, row))
            tgt_ids = tok.encode(tgt)
            all_tgt_ids.extend(tgt_ids)

    print(f"Total valid pairs loaded: {len(pairs):,}")
    
    if base_maps_path and Path(base_maps_path).exists():
        print(f"Loading existing base edit maps from {base_maps_path}...")
        with Path(base_maps_path).open("r", encoding="utf-8") as f:
            base_payload = json.load(f)
        replace_map = {int(k): v for k, v in base_payload["replace"].items()}
        append_map = {int(k): v for k, v in base_payload["append"].items()}
        maps_payload = base_payload
    else:
        print(f"Building dynamic edit maps (k_replace={k_replace}, k_append={k_append})...")
        replace_map, append_map = build_freq_maps(all_tgt_ids, k_replace=k_replace, k_append=k_append)
        maps_payload = {
            "replace": {str(k): v for k, v in replace_map.items()},
            "append": {str(k): v for k, v in append_map.items()},
            "id_to_replace": {str(v): int(k) for k, v in replace_map.items()},
            "id_to_append": {str(v): int(k) for k, v in append_map.items()},
        }

    maps_path.write_text(json.dumps(maps_payload, indent=2), encoding="utf-8")
    print(f"Wrote edit maps -> {maps_path}")

    print(f"Aligning tokens and generating edit tag sequences -> {out_jsonl}...")

    def generate_rows():
        for src, tgt, meta in pairs:
            src_ids = tok.encode(src)
            tgt_ids = tok.encode(tgt)
            src_toks = tok.token_strings(src)
            tgt_toks = tok.token_strings(tgt)

            labels = align_edits(
                src_ids,
                tgt_ids,
                replace_map,
                append_map,
                src_pieces=src_toks,
                tgt_pieces=tgt_toks,
            )
            chars = tok.char_ids_for_tokens(src_toks)
            n = min(len(src_ids), len(labels), len(chars))

            yield {
                "id": meta.get("id", ""),
                "src": src,
                "clean": tgt,
                "src_ids": src_ids[:n],
                "src_chars": chars[:n],
                "edit_labels": labels[:n],
                "error_type": meta.get("error_type", ""),
                "error_category": meta.get("error_category", ""),
                "domain": meta.get("domain", ""),
            }

    n_written = write_jsonl(out_jsonl, generate_rows())
    print(f"Successfully converted {n_written:,} rows to token-aligned GEC JSONL -> {out_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV into TypoFormer GEC JSONL format.")
    parser.add_argument("--csv", default="data/dataset/wine_train.csv", help="Input CSV path")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json", help="BPE Tokenizer path")
    parser.add_argument("--out", default="data/processed/gec/v5/train.jsonl", help="Output JSONL path")
    parser.add_argument("--maps", default="data/processed/gec/v5/edit_maps.json", help="Output edit maps JSON path")
    parser.add_argument("--base-maps", default=None, help="Optional existing edit maps JSON to preserve class mappings")
    parser.add_argument("--k-replace", type=int, default=2048)
    parser.add_argument("--k-append", type=int, default=2048)
    args = parser.parse_args()

    convert_csv_to_gec(
        csv_path=args.csv,
        tokenizer_path=args.tokenizer,
        out_jsonl=args.out,
        maps_path=args.maps,
        base_maps_path=args.base_maps,
        k_replace=args.k_replace,
        k_append=args.k_append,
    )


if __name__ == "__main__":
    main()
