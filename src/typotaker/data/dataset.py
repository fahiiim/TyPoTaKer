from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from typotaker.data.jsonl import iter_jsonl
from typotaker.tokenizer import TypoTokenizer


class PackedLMDataset(IterableDataset):
    def __init__(
        self,
        paths: list[str],
        tokenizer: TypoTokenizer,
        seq_len: int,
        skip_lines: int = 0,
    ) -> None:
        self.paths = [str(Path(p)) for p in paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.skip_lines = skip_lines

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf_tok: list[int] = []
        buf_txt: list[str] = []
        skipped = 0
        while True:
            progressed = False
            for path in self.paths:
                for row in iter_jsonl(path):
                    if skipped < self.skip_lines:
                        skipped += 1
                        continue
                    text = row.get("text") or row.get("clean") or ""
                    if not text.strip():
                        continue
                    progressed = True
                    ids = self.tokenizer.encode(text)
                    pieces = self.tokenizer.token_strings(text)
                    buf_tok.extend(ids)
                    buf_txt.extend(pieces)
                    while len(buf_tok) >= self.seq_len:
                        tok = buf_tok[: self.seq_len]
                        pcs = buf_txt[: self.seq_len]
                        buf_tok = buf_tok[self.seq_len :]
                        buf_txt = buf_txt[self.seq_len :]
                        chars = self.tokenizer.char_ids_for_tokens(pcs)
                        yield {
                            "token_ids": torch.tensor(tok, dtype=torch.long),
                            "char_ids": torch.tensor(chars, dtype=torch.long),
                        }
            if not progressed and skipped >= self.skip_lines:
                raise RuntimeError(f"no LM text found in {self.paths}")
            # after the first pass, skip_lines only applied once
            self.skip_lines = 0
            skipped = 0


class GECDataset(IterableDataset):
    def __init__(
        self,
        paths: list[str],
        tokenizer: TypoTokenizer,
        seq_len: int,
        skip_lines: int = 0,
    ) -> None:
        self.paths = [str(Path(p)) for p in paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.skip_lines = skip_lines

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        skipped = 0
        pad = self.tokenizer.pad_id
        while True:
            progressed = False
            for path in self.paths:
                for row in iter_jsonl(path):
                    if skipped < self.skip_lines:
                        skipped += 1
                        continue
                    progressed = True
                    ids = row["src_ids"][: self.seq_len]
                    chars = row["src_chars"][: self.seq_len]
                    labels = row["edit_labels"][: self.seq_len]
                    t = len(ids)
                    if t < self.seq_len:
                        pad_n = self.seq_len - t
                        ids = ids + [pad] * pad_n
                        chars = chars + [[0] * self.tokenizer.max_chars] * pad_n
                        labels = labels + [-100] * pad_n
                    yield {
                        "token_ids": torch.tensor(ids, dtype=torch.long),
                        "char_ids": torch.tensor(chars, dtype=torch.long),
                        "edit_labels": torch.tensor(labels, dtype=torch.long),
                    }
            if not progressed and skipped >= self.skip_lines:
                raise RuntimeError(f"no GEC rows found in {self.paths}")
            self.skip_lines = 0
            skipped = 0


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
