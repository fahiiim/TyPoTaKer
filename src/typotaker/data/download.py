"""Download and mix pretraining text. Quality-filtered; no GEC labels yet."""

from __future__ import annotations

import json
from pathlib import Path

from typotaker.data.filter import chunk_document, looks_like_english


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("pip install 'typotaker[data]'") from exc
    return load_dataset


def _write_rows(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 5000 == 0:
                print(f"  ... {n} rows", flush=True)
    return n


def iter_tinystories(max_rows: int):
    load_dataset = _require_datasets()
    ds = load_dataset("roneneldan/TinyStories", split="train")
    kept = 0
    for row in ds:
        if kept >= max_rows:
            break
        text = (row.get("text") or "").strip()
        if looks_like_english(text, min_chars=60, max_chars=4000):
            yield {"text": text, "source": "tinystories"}
            kept += 1


def iter_fineweb_edu(max_chunks: int):
    load_dataset = _require_datasets()
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    kept = 0
    for row in ds:
        if kept >= max_chunks:
            break
        text = (row.get("text") or "").strip()
        for chunk in chunk_document(text):
            yield {"text": chunk, "source": "fineweb-edu"}
            kept += 1
            if kept >= max_chunks:
                break


def build_pretrain_corpus(
    out_jsonl: str,
    *,
    tinystories: int = 80_000,
    fineweb_chunks: int = 60_000,
) -> str:
    out = Path(out_jsonl)
    print(f"[data] TinyStories up to {tinystories}", flush=True)
    stories = list(iter_tinystories(tinystories))
    print(f"[data] TinyStories kept {len(stories)}", flush=True)
    print(f"[data] FineWeb-Edu up to {fineweb_chunks} chunks", flush=True)
    web = list(iter_fineweb_edu(fineweb_chunks))
    print(f"[data] FineWeb-Edu kept {len(web)}", flush=True)
    mixed = stories + web
    n = _write_rows(out, iter(mixed))
    print(f"[data] wrote {n} rows -> {out}")
    return str(out)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed/lm/train.jsonl")
    p.add_argument("--tinystories", type=int, default=80_000)
    p.add_argument("--fineweb-chunks", type=int, default=60_000)
    args = p.parse_args()
    build_pretrain_corpus(args.out, tinystories=args.tinystories, fineweb_chunks=args.fineweb_chunks)
