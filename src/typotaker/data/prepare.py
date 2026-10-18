from __future__ import annotations

import json
import random
from pathlib import Path

from typotaker.data.edits import align_edits, build_freq_maps
from typotaker.data.errors import corrupt_sentence
from typotaker.data.jsonl import write_jsonl
from typotaker.tokenizer import TypoTokenizer, train_bpe

SMOKE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "FastAPI makes it simple to build high performance APIs.",
    "Please check their configuration before you deploy.",
    "It's important that the model does not change correct text.",
    "She goes to the store and buys apples.",
    "There are many ways to design a small language model.",
    "You're going to resume training from the last checkpoint.",
    "The server listens on port 8000 by default.",
    "Grammar correction should prefer precision over recall.",
    "Next word prediction helps while you type on the desktop.",
] * 80


def write_smoke_corpus(root: Path) -> Path:
    path = root / "data" / "processed" / "lm" / "smoke.jsonl"
    n = write_jsonl(path, ({"text": s} for s in SMOKE_SENTENCES))
    print(f"wrote {n} smoke LM rows -> {path}")
    return path


def train_tokenizer_from_jsonl(jsonl_paths: list[str], out: str, vocab_size: int) -> str:
    texts = Path("data/processed/_tok_corpus.txt")
    texts.parent.mkdir(parents=True, exist_ok=True)
    with texts.open("w", encoding="utf-8") as f:
        for p in jsonl_paths:
            with Path(p).open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    f.write((obj.get("text") or obj.get("clean") or "") + "\n")
    return train_bpe([str(texts)], out, vocab_size=vocab_size)


def build_gec_from_clean(
    clean_jsonl: str,
    tokenizer_path: str,
    out_jsonl: str,
    *,
    k_replace: int = 2048,
    k_append: int = 2048,
    seed: int = 42,
    max_chars: int = 16,
) -> None:
    tok = TypoTokenizer(tokenizer_path, max_chars=max_chars)
    rng = random.Random(seed)
    clean_rows = []
    all_ids: list[int] = []
    with Path(clean_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("clean") or ""
            if not text.strip():
                continue
            ids = tok.encode(text)
            clean_rows.append(text)
            all_ids.extend(ids)
    replace_map, append_map = build_freq_maps(all_ids, k_replace, k_append)
    maps_path = Path(out_jsonl).with_name("edit_maps.json")
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.write_text(
        json.dumps(
            {
                "replace": {str(k): v for k, v in replace_map.items()},
                "append": {str(k): v for k, v in append_map.items()},
                "id_to_replace": {str(v): int(k) for k, v in replace_map.items()},
                "id_to_append": {str(v): int(k) for k, v in append_map.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def rows():
        for text in clean_rows:
            noisy = corrupt_sentence(text, rng)
            src_ids = tok.encode(noisy)
            tgt_ids = tok.encode(text)
            src_toks = tok.token_strings(noisy)
            labels = align_edits(src_ids, tgt_ids, replace_map, append_map)
            chars = tok.char_ids_for_tokens(src_toks)
            n = min(len(src_ids), len(labels), len(chars))
            yield {
                "src": noisy,
                "clean": text,
                "src_ids": src_ids[:n],
                "src_chars": chars[:n],
                "edit_labels": labels[:n],
            }

    n = write_jsonl(out_jsonl, rows())
    print(f"wrote {n} GEC pairs -> {out_jsonl}")
    print(f"edit maps -> {maps_path}")
