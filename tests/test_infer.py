from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from typotaker.config import ModelConfig
from typotaker.infer import next_words, correct_sentence
from typotaker.models.typoformer import TyPoFormer
from typotaker.tokenizer import TypoTokenizer


@pytest.fixture
def tiny_system(tmp_path: Path):
    tok_path = Path("data/tokenizer/tokenizer.json")
    if not tok_path.exists():
        pytest.skip("data/tokenizer/tokenizer.json not found")

    tok = TypoTokenizer(str(tok_path), max_chars=16)
    
    cfg = ModelConfig(
        vocab_size=len(tok.tok.get_vocab()),
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=64,
        char_dim=16,
        n_replace_classes=16,
        n_append_classes=16,
    )
    model = TyPoFormer(cfg).eval()

    # Create dummy edit maps
    maps_file = tmp_path / "edit_maps.json"
    dummy_maps = {
        "id_to_replace": {"4": 100, "5": 101},
        "id_to_append": {"20": 102},
    }
    maps_file.write_text(json.dumps(dummy_maps), encoding="utf-8")

    return model, tok, str(maps_file)


def test_next_words_inference(tiny_system):
    model, tok, _ = tiny_system
    prompt = "The server is"
    k = 3
    suggestions = next_words(model, tok, prompt, k=k)

    assert len(suggestions) == k
    for word, prob in suggestions:
        assert isinstance(word, str)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


def test_correct_sentence_inference(tiny_system):
    model, tok, maps_path = tiny_system
    text = "This is a test sentence."
    corrected = correct_sentence(
        model,
        tok,
        text,
        maps_path=maps_path,
        min_edit_prob=0.0,
        edit_margin=0.0,
    )
    assert isinstance(corrected, str)
    assert len(corrected) > 0


def test_confidence_gated_correction(tiny_system):
    model, tok, maps_path = tiny_system
    text = "This is a test sentence."
    # High threshold (0.999) should force KEEP for almost any random untrained model output
    gated_corrected = correct_sentence(
        model,
        tok,
        text,
        maps_path=maps_path,
        min_edit_prob=0.999,
        edit_margin=0.5,
    )
    assert isinstance(gated_corrected, str)
