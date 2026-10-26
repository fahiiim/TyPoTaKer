from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


@dataclass
class ModelConfig:
    name: str = "typoformer_stage0_25m"
    architecture: str = "typoformer"
    vocab_size: int = 16000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    char_vocab_size: int = 128
    char_dim: int = 32
    max_chars_per_token: int = 16
    n_replace_classes: int = 2048
    n_append_classes: int = 2048
    char_gate_init: float = 0.1
    qk_norm: bool = True
    n_mtp: int = 2
    grow_from_stage: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        return cls(**_load_yaml(path))

    def n_edit_labels(self) -> int:
        # KEEP, DELETE, CASE_CAP, CASE_LOWER, then REPLACE_* and APPEND_*
        return 4 + self.n_replace_classes + self.n_append_classes


@dataclass
class DataConfig:
    train_paths: list[str] = field(default_factory=list)
    val_paths: list[str] = field(default_factory=list)
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    seq_len: int = 256
    pack_sequences: bool = True
    preload: bool = True


@dataclass
class OptimConfig:
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 20_000
    grad_clip: float = 1.0


@dataclass
class TrainLoopConfig:
    micro_batch_size: int = 16
    grad_accum: int = 4
    amp_dtype: str = "bf16"
    compile: bool = True
    compile_mode: str = "max-autotune"
    seed: int = 42
    max_hours: float = 6.0
    log_every: int = 20
    eval_every: int = 100
    save_every_steps: int = 500
    keep_last: int = 4
    checkpoint_dir: str = "checkpoints"
    resume: str = "auto"
    num_workers: int = 4
    keep_downweight: float = 1.0


@dataclass
class TrainConfig:
    task: str = "lm"
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    init_from: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw = _load_yaml(path)
        model_ref = raw.get("model", {})
        if isinstance(model_ref, str):
            model = ModelConfig.from_yaml(model_ref)
        else:
            model = ModelConfig(**model_ref)
        data = DataConfig(**raw.get("data", {}))
        opt_raw = dict(raw.get("optim", {}))
        if "betas" in opt_raw:
            opt_raw["betas"] = tuple(opt_raw["betas"])
        optim = OptimConfig(**opt_raw)
        loop = TrainLoopConfig(**raw.get("train", {}))
        return cls(
            task=raw.get("task", "lm"),
            run_name=raw.get("run_name", "run"),
            model=model,
            data=data,
            optim=optim,
            train=loop,
            init_from=raw.get("init_from"),
        )
