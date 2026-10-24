from __future__ import annotations

from pathlib import Path

import pytest

from typotaker.config import ModelConfig, TrainConfig


def test_default_model_config():
    cfg = ModelConfig()
    assert cfg.d_model % cfg.n_heads == 0
    assert cfg.n_heads % cfg.n_kv_heads == 0
    assert cfg.n_edit_labels() == 4 + cfg.n_replace_classes + cfg.n_append_classes


def test_load_all_model_configs():
    model_configs_dir = Path("configs/model")
    if not model_configs_dir.exists():
        pytest.skip("configs/model directory does not exist")

    yaml_files = list(model_configs_dir.glob("*.yaml"))
    assert len(yaml_files) > 0, "No model config YAML files found"

    for yaml_path in yaml_files:
        cfg = ModelConfig.from_yaml(yaml_path)
        assert cfg.d_model > 0
        assert cfg.n_layers > 0
        assert cfg.n_heads > 0
        assert cfg.n_kv_heads > 0
        assert cfg.d_model % cfg.n_heads == 0, f"{yaml_path}: d_model must divide by n_heads"
        assert cfg.n_heads % cfg.n_kv_heads == 0, f"{yaml_path}: n_heads must divide by n_kv_heads"
        assert cfg.n_edit_labels() == 4 + cfg.n_replace_classes + cfg.n_append_classes


def test_load_all_train_configs():
    train_configs_dir = Path("configs/train")
    if not train_configs_dir.exists():
        pytest.skip("configs/train directory does not exist")

    yaml_files = list(train_configs_dir.glob("*.yaml"))
    assert len(yaml_files) > 0, "No train config YAML files found"

    for yaml_path in yaml_files:
        train_cfg = TrainConfig.from_yaml(yaml_path)
        assert train_cfg.task in ["lm", "gec", "joint"]
        assert train_cfg.optim.lr > 0
        assert train_cfg.train.micro_batch_size > 0
        assert train_cfg.train.grad_accum >= 1
