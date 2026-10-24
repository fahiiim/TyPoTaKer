from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from typotaker.data.dataset import PackedLMDataset, GECDataset
from typotaker.tokenizer import TypoTokenizer


@pytest.fixture
def dummy_tokenizer(tmp_path: Path):
    repo_tok = Path("data/tokenizer/tokenizer.json")
    if repo_tok.exists():
        return TypoTokenizer(str(repo_tok), max_chars=16)
    
    # Otherwise fallback
    tok_file = tmp_path / "tok.json"
    from typotaker.tokenizer import train_bpe
    corpus = tmp_path / "c.txt"
    corpus.write_text("Hello world testing datasets and tokenizers.", encoding="utf-8")
    train_bpe([str(corpus)], str(tok_file), vocab_size=128)
    return TypoTokenizer(str(tok_file), max_chars=16)


def test_packed_lm_dataset(dummy_tokenizer, tmp_path: Path):
    data_file = tmp_path / "lm_data.jsonl"
    with data_file.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"text": f"This is sentence number {i} for language modeling testing."}) + "\n")

    seq_len = 16
    ds = PackedLMDataset(paths=[str(data_file)], tokenizer=dummy_tokenizer, seq_len=seq_len)
    it = iter(ds)
    batch = next(it)

    assert "token_ids" in batch
    assert "char_ids" in batch
    assert batch["token_ids"].shape == (seq_len,)
    assert batch["char_ids"].shape == (seq_len, dummy_tokenizer.max_chars)
    assert batch["token_ids"].dtype == torch.long
    assert batch["char_ids"].dtype == torch.long


def test_gec_dataset_padding(dummy_tokenizer, tmp_path: Path):
    data_file = tmp_path / "gec_data.jsonl"
    # Row with short sequence (length 4) to test padding up to seq_len 16
    row = {
        "src_ids": [10, 20, 30, 40],
        "src_chars": [[1] * dummy_tokenizer.max_chars for _ in range(4)],
        "edit_labels": [0, 1, 4, 0],
    }
    with data_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    seq_len = 16
    ds = GECDataset(paths=[str(data_file)], tokenizer=dummy_tokenizer, seq_len=seq_len)
    it = iter(ds)
    batch = next(it)

    assert batch["token_ids"].shape == (seq_len,)
    assert batch["char_ids"].shape == (seq_len, dummy_tokenizer.max_chars)
    assert batch["edit_labels"].shape == (seq_len,)
    
    # Padded positions in edit_labels must be -100 (standard PyTorch ignore_index)
    assert (batch["edit_labels"][4:] == -100).all()
    # Padded positions in token_ids must equal pad_id
    assert (batch["token_ids"][4:] == dummy_tokenizer.pad_id).all()
