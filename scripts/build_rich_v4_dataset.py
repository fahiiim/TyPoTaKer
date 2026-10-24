"""Generate rich multi-domain v4 GEC & LM dataset.

Combines FineWeb-Edu clean text with rich conversational, desktop, and daily
writing patterns, running the upgraded 10-family error generation engine.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from typotaker.data.edits import align_edits, build_freq_maps
from typotaker.data.errors import corrupt_sentence
from typotaker.data.filter import chunk_document, looks_like_english
from typotaker.data.prepare import split_gec_units
from typotaker.tokenizer import TypoTokenizer

CURATED_CONVERSATIONAL = [
    # Everyday questions and work requests
    "Could you please review the attached document before the meeting starts?",
    "Let me know if you need any additional information from my side.",
    "I am looking forward to working with you on this upcoming project.",
    "Please find the latest version of the configuration file attached below.",
    "I will be working remotely today but I am available on Slack and email.",
    "We need to schedule a follow up discussion regarding the database migration.",
    "Thank you so much for your quick response and detailed explanation.",
    "Can you clarify what the expected output should be for this test case?",
    "I have updated the pull request according to your feedback.",
    "The server listening on port 8080 experienced an unexpected timeout.",
    "Make sure to save your work before restarting your computer.",
    "It is important that we maintain consistent naming conventions across the codebase.",
    "Let us know when you have finished testing the new feature.",
    "I appreciate your patience while we investigate this intermittent bug.",
    "Please do not hesitate to contact our support team if you have any questions.",
    "She has been working on this problem since early this morning.",
    "They were unable to reproduce the crash on their local machine.",
    "Everyone is invited to attend the company all-hands meeting tomorrow afternoon.",
    "One of the major advantages of an offline assistant is privacy and speed.",
    "You just want to make me happy when it happens so unexpectedly.",
    "The quick brown fox jumps over the lazy dog in the sunny park.",
    "I would like to know whether the delivery address has been verified.",
    "Their decision to adopt modern language models improved overall productivity.",
    "We definitely need to check if the network connection is stable.",
    "He does not believe that the error occurred because of the recent update.",
    "She goes to the grocery store every Sunday to buy fresh fruit and vegetables.",
    "It seems that the cache was cleared automatically during the build process.",
    "Between you and me, I think this design is much cleaner and easier to maintain.",
    "They should have notified us before making breaking changes to the API.",
    "Could you please check whether the package has been delivered to your office?",
]


def stream_fineweb_text(target_count: int = 80_000):
    from datasets import load_dataset
    print(f"[corpus] Streaming FineWeb-Edu up to {target_count} clean chunks...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    kept = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text or len(text) < 80:
            continue
        if looks_like_english(text, min_chars=60, max_chars=4000):
            for chunk in chunk_document(text, target=600):
                yield chunk
                kept += 1
                if kept % 10_000 == 0:
                    print(f"  ... {kept}/{target_count} FineWeb chunks collected", flush=True)
                if kept >= target_count:
                    return


def build_rich_v4(
    out_gec_jsonl: str = "data/processed/gec/v4/train.jsonl",
    out_lm_jsonl: str = "data/processed/lm/train_v4.jsonl",
    tokenizer_path: str = "data/tokenizer/tokenizer.json",
    target_gec_units: int = 500_000,
    k_replace: int = 4096,
    k_append: int = 2048,
    seed: int = 42,
):
    t0 = time.time()
    rng = random.Random(seed)
    tok = TypoTokenizer(tokenizer_path, max_chars=16)

    print(f"\n=== Building Rich Multi-Domain v4 Dataset (Target: {target_gec_units:,} units) ===")

    # 1. Collect clean documents from multiple sources
    clean_docs = []
    
    # Existing LM train text if available
    existing_lm = Path("data/processed/lm/train.jsonl")
    if existing_lm.exists():
        print(f"[corpus] Reading existing clean text from {existing_lm}...")
        with existing_lm.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("text") or obj.get("clean") or ""
                    if len(txt) > 40:
                        clean_docs.append(txt)
                        if len(clean_docs) >= 50_000:
                            break
                except Exception:
                    pass
        print(f"[corpus] Loaded {len(clean_docs)} baseline documents.")

    # Add curated conversational sentences (repeated with variation)
    print(f"[corpus] Adding {len(CURATED_CONVERSATIONAL) * 500} conversational anchors...")
    for _ in range(500):
        clean_docs.extend(CURATED_CONVERSATIONAL)

    # Stream FineWeb-Edu clean text to reach target breadth
    needed = max(0, 120_000 - len(clean_docs))
    if needed > 0:
        try:
            for chunk in stream_fineweb_text(needed):
                clean_docs.append(chunk)
                if len(clean_docs) >= 120_000:
                    break
        except Exception as e:
            print(f"[warn] Could not stream additional FineWeb text ({e}); using local corpus")

    print(f"[corpus] Total clean documents gathered: {len(clean_docs):,}")
    rng.shuffle(clean_docs)

    # 2. Extract sentence units and build vocabulary frequency maps
    print("[corpus] Extracting sentence units and building frequency maps...")
    units: list[str] = []
    all_clean_ids: list[int] = []

    for doc in clean_docs:
        doc_units = split_gec_units(doc, max_chars=320)
        for u in doc_units:
            u_clean = u.strip()
            if 15 <= len(u_clean) <= 320:
                units.append(u_clean)
                ids = tok.encode(u_clean, add_special=False)
                all_clean_ids.extend(ids)
                if len(units) >= target_gec_units:
                    break
        if len(units) >= target_gec_units:
            break

    print(f"[corpus] Gathered {len(units):,} clean sentence units ({len(all_clean_ids):,} total tokens).")

    # 3. Build & save edit maps
    rep_map, app_map = build_freq_maps(all_clean_ids, k_replace=k_replace, k_append=k_append)
    maps_path = Path(out_gec_jsonl).with_name("edit_maps.json")
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.write_text(
        json.dumps(
            {
                "replace": {str(k): v for k, v in rep_map.items()},
                "append": {str(k): v for k, v in app_map.items()},
                "id_to_replace": {str(v): int(k) for k, v in rep_map.items()},
                "id_to_append": {str(v): int(k) for k, v in app_map.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[maps] Saved edit maps (replace={len(rep_map)}, append={len(app_map)}) -> {maps_path}")

    # 4. Save enhanced LM training corpus
    Path(out_lm_jsonl).parent.mkdir(parents=True, exist_ok=True)
    print(f"[lm] Writing enriched LM corpus -> {out_lm_jsonl}...")
    with Path(out_lm_jsonl).open("w", encoding="utf-8") as f_lm:
        for u in units:
            f_lm.write(json.dumps({"text": u}) + "\n")

    # 5. Generate paired GEC rows with upgraded 10-family noise
    print(f"[gec] Generating GEC paired dataset -> {out_gec_jsonl}...")
    written = 0
    with Path(out_gec_jsonl).open("w", encoding="utf-8") as f_gec:
        for idx, clean_unit in enumerate(units):
            # 70% corrupted, 30% pristine clean (KEEP anchors)
            is_corrupt = rng.random() < 0.70
            if is_corrupt:
                noisy = corrupt_sentence(
                    clean_unit,
                    rng,
                    p=1.0,
                    min_ops=1,
                    max_ops=3,
                )
            else:
                noisy = clean_unit

            src_ids = tok.encode(noisy, add_special=True)
            tgt_ids = tok.encode(clean_unit, add_special=True)
            src_toks = tok.token_strings(noisy, add_special=True)
            tgt_toks = tok.token_strings(clean_unit, add_special=True)

            labels = align_edits(
                src_ids,
                tgt_ids,
                rep_map,
                app_map,
                src_pieces=src_toks,
                tgt_pieces=tgt_toks,
            )
            chars = tok.char_ids_for_tokens(src_toks)
            n = min(len(src_ids), len(labels), len(chars))

            row = {
                "src": noisy,
                "clean": clean_unit,
                "src_ids": src_ids[:n],
                "src_chars": chars[:n],
                "edit_labels": labels[:n],
            }
            f_gec.write(json.dumps(row) + "\n")
            written += 1

            if written % 50_000 == 0:
                print(f"  ... {written:,}/{len(units):,} GEC units processed", flush=True)

    elapsed = time.time() - t0
    print(f"\n[DONE] Successfully built rich v4 dataset in {elapsed:.1f}s ({elapsed/60:.1f} mins)!")
    print(f"  LM Corpus:  {out_lm_jsonl} ({written:,} sentences)")
    print(f"  GEC Corpus: {out_gec_jsonl} ({written:,} paired examples)")
    print(f"  Edit Maps:  {maps_path}")


if __name__ == "__main__":
    build_rich_v4()
