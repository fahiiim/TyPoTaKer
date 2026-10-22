"""Generate Stage 2 GEC training dataset with full QWERTY noise and 4096 edit classes."""
import time
from pathlib import Path
from typotaker.data.prepare import build_gec_from_clean

clean_jsonl = "data/processed/gec/clean_subset.jsonl"
tokenizer_path = "data/tokenizer/tokenizer.json"
out_jsonl = "data/processed/gec/v3/train.jsonl"

print(f"Generating Stage 2 GEC dataset from {clean_jsonl}...")
t0 = time.time()

build_gec_from_clean(
    clean_jsonl=clean_jsonl,
    tokenizer_path=tokenizer_path,
    out_jsonl=out_jsonl,
    k_replace=4096,
    k_append=2048,
    seed=42,
    max_chars=16,
    corruption_prob=0.80,
    min_ops=1,
    max_ops=2,
    max_unit_chars=384,
)

elapsed = time.time() - t0
print(f"Dataset generated in {elapsed:.1f}s -> {out_jsonl}")
