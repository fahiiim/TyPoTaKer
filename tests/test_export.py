from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from typotaker.config import ModelConfig
from typotaker.export import export_onnx, quantize_onnx
from typotaker.models.typoformer import TyPoFormer
from typotaker.train.checkpoint import save_checkpoint


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        name="unit_export",
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=16,
        char_dim=8,
        n_replace_classes=4,
        n_append_classes=4,
    )


def test_export_and_quantize_onnx():
    cfg = _tiny_config()
    model = TyPoFormer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_root = tmp_path / "ckpt"

        save_checkpoint(
            root=ckpt_root,
            step=1,
            samples_seen=10,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            model_cfg=cfg,
        )

        onnx_out = tmp_path / "model.onnx"
        int8_out = tmp_path / "model_int8.onnx"

        # 1. Export ONNX (LM)
        exported = export_onnx(str(ckpt_root), str(onnx_out), task="lm", seq_len=16)
        assert Path(exported).exists()
        assert Path(exported).stat().st_size > 0

        # 2. Quantize ONNX (INT8)
        quantized = quantize_onnx(exported, str(int8_out))
        assert Path(quantized).exists()
        assert Path(quantized).stat().st_size > 0

        # 3. Verify ONNX Runtime inference
        sess = ort.InferenceSession(quantized, providers=["CPUExecutionProvider"])
        tokens = np.ones((1, 8), dtype=np.int64)
        chars = np.zeros((1, 8, cfg.max_chars_per_token), dtype=np.int64)
        outputs = sess.run(None, {"token_ids": tokens, "char_ids": chars})

        # Expected output shape: (batch, seq, vocab_size)
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 8, cfg.vocab_size)
