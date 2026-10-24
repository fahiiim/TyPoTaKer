from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from typotaker.tokenizer import TypoTokenizer, train_bpe, SPECIAL, PAD, UNK, BOS, EOS


@pytest.fixture(scope="module")
def sample_tokenizer_path():
    # If the project's trained tokenizer exists, use it; otherwise train a tiny one
    repo_tok = Path("data/tokenizer/tokenizer.json")
    if repo_tok.exists():
        yield str(repo_tok)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = Path(tmpdir) / "corpus.txt"
        corpus.write_text("The quick brown fox jumps over the lazy dog.\nHello world from TyPoTaKer!", encoding="utf-8")
        out_path = Path(tmpdir) / "tokenizer.json"
        train_bpe([str(corpus)], str(out_path), vocab_size=256)
        yield str(out_path)


def test_tokenizer_special_tokens(sample_tokenizer_path):
    tok = TypoTokenizer(sample_tokenizer_path, max_chars=16)
    assert tok.pad_id is not None
    assert tok.unk_id is not None
    assert tok.bos_id is not None
    assert tok.eos_id is not None
    assert len({tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id}) == 4


def test_tokenizer_encode_decode_roundtrip(sample_tokenizer_path):
    tok = TypoTokenizer(sample_tokenizer_path, max_chars=16)
    text = "The quick brown fox jumps over the lazy dog."
    
    # With special tokens
    ids_special = tok.encode(text, add_special=True)
    assert ids_special[0] == tok.bos_id
    assert ids_special[-1] == tok.eos_id
    assert len(ids_special) > 2

    # Without special tokens
    ids_raw = tok.encode(text, add_special=False)
    assert tok.bos_id not in ids_raw
    assert tok.eos_id not in ids_raw

    # Roundtrip decode
    decoded = tok.decode(ids_special)
    assert decoded.strip() == text.strip()


def test_tokenizer_token_strings(sample_tokenizer_path):
    tok = TypoTokenizer(sample_tokenizer_path, max_chars=16)
    text = "Hello world"
    pieces = tok.token_strings(text, add_special=True)
    assert pieces[0] == BOS
    assert pieces[-1] == EOS
    assert len(pieces) >= 3


def test_tokenizer_char_ids_matrix(sample_tokenizer_path):
    max_chars = 16
    tok = TypoTokenizer(sample_tokenizer_path, max_chars=max_chars)
    tokens = ["Hello", "world", "supercalifragilisticexpialidocious"]
    chars = tok.char_ids_for_tokens(tokens)

    assert len(chars) == len(tokens)
    for row in chars:
        assert len(row) == max_chars
        # All byte values should be in ASCII byte range [0, 127]
        for b in row:
            assert 0 <= b <= 127

    # Test truncation on long tokens
    long_row = chars[2]
    assert len(long_row) == max_chars
    # Should not have all zeros
    assert any(b != 0 for b in long_row)


def test_tokenizer_empty_and_whitespace(sample_tokenizer_path):
    tok = TypoTokenizer(sample_tokenizer_path, max_chars=16)
    
    # Empty string
    ids_empty = tok.encode("", add_special=False)
    assert ids_empty == []

    # Whitespace decode
    ids_ws = tok.encode("   ", add_special=False)
    assert isinstance(ids_ws, list)
    dec = tok.decode(ids_ws)
    assert dec == "   "
