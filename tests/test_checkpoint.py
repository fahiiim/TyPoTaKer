from pathlib import Path

import torch

from typotaker.config import ModelConfig
from typotaker.models.typoformer import TyPoFormer
from typotaker.train.checkpoint import find_latest, load_checkpoint, save_checkpoint


def test_roundtrip(tmp_path: Path):
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, ffn_hidden=64)
    model = TyPoFormer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        root=tmp_path / "run",
        step=12,
        samples_seen=99,
        model=model,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        model_cfg=cfg,
        extra={"task": "lm"},
        keep_last=2,
    )
    latest = find_latest(tmp_path / "run")
    assert latest is not None
    blob = load_checkpoint(latest)
    assert blob["meta"]["step"] == 12
    assert blob["meta"]["samples_seen"] == 99
    clone = TyPoFormer(cfg)
    clone.load_state_dict(blob["model"])
    for a, b in zip(model.parameters(), clone.parameters()):
        assert torch.equal(a, b)
