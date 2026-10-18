from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from typotaker.config import TrainConfig
from typotaker.data.dataset import GECDataset, PackedLMDataset, collate
from typotaker.models.growth import grow_model
from typotaker.models.typoformer import TyPoFormer, count_parameters
from typotaker.tokenizer import TypoTokenizer
from typotaker.train.checkpoint import (
    find_latest,
    load_checkpoint,
    restore_rng,
    run_dir,
    save_checkpoint,
)
from typotaker.train.schedule import cosine_with_warmup


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def amp_dtype(name: str) -> torch.dtype | None:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return None


def build_optimizer(model: TyPoFormer, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith("bias") or "norm" in n.lower() or "gate" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.optim.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        fused=fused,
    )


def maybe_compile(model: TyPoFormer, cfg: TrainConfig) -> TyPoFormer:
    if not cfg.train.compile or not torch.cuda.is_available():
        return model
    try:
        return torch.compile(model, mode=cfg.train.compile_mode)  # type: ignore[return-value]
    except Exception as exc:
        print(f"[warn] torch.compile disabled: {exc}")
        return model


def _maybe_init_from(model: TyPoFormer, init_from: str | None, device: torch.device) -> TyPoFormer:
    if not init_from:
        return model
    path = Path(init_from)
    if path.is_dir() or path.name in {"latest", "latest.json"}:
        root = path.parent if path.name in {"latest", "latest.json"} else path
        latest = find_latest(root)
        if latest is None:
            print(f"[warn] init_from {init_from} has no checkpoint; training from scratch")
            return model
        path = latest
    if path.is_file():
        path = path.parent
    if not (path / "model.pt").exists():
        print(f"[warn] no model.pt at {path}")
        return model
    blob = load_checkpoint(path)
    src_cfg = type(model.cfg)(**blob["meta"]["model"])
    src = TyPoFormer(src_cfg)
    src.load_state_dict(blob["model"], strict=False)
    if (src.cfg.n_layers, src.cfg.d_model) != (model.cfg.n_layers, model.cfg.d_model):
        print(f"[grow] {src.cfg.name} -> {model.cfg.name}")
        model = grow_model(src, model.cfg)
    else:
        model.load_state_dict(src.state_dict(), strict=False)
    return model.to(device)


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.train.seed)
    device = pick_device()
    print(f"device={device} cuda={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    tokenizer = TypoTokenizer(cfg.data.tokenizer_path, max_chars=cfg.model.max_chars_per_token)
    root = run_dir(cfg.train.checkpoint_dir, cfg.run_name)
    root.mkdir(parents=True, exist_ok=True)

    model = TyPoFormer(cfg.model).to(device)
    model = _maybe_init_from(model, cfg.init_from, device)
    print(f"params={count_parameters(model):,}  task={cfg.task}  run={cfg.run_name}")

    optimizer = build_optimizer(model, cfg)
    scheduler = cosine_with_warmup(optimizer, cfg.optim.warmup_steps, cfg.optim.max_steps)
    dtype = amp_dtype(cfg.train.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and device.type == "cuda")

    start_step = 0
    samples_seen = 0
    resume_path = None
    if cfg.train.resume == "auto":
        resume_path = find_latest(root)
    elif cfg.train.resume not in {"", "none", "false"}:
        resume_path = Path(cfg.train.resume)

    if resume_path is not None:
        blob = load_checkpoint(resume_path)
        model.load_state_dict(blob["model"], strict=True)
        grown = bool(blob["meta"].get("grown_from"))
        if not grown:
            optimizer.load_state_dict(blob["optimizer"])
            if "scheduler" in blob:
                scheduler.load_state_dict(blob["scheduler"])
            if "scaler" in blob and scaler is not None:
                scaler.load_state_dict(blob["scaler"])
            restore_rng(blob)
        else:
            print("[resume] grown checkpoint: fresh optimizer/scheduler")
        start_step = int(blob["meta"]["step"])
        samples_seen = int(blob["meta"].get("samples_seen", 0))
        print(f"[resume] {resume_path} step={start_step} samples_seen={samples_seen}")

    skip = samples_seen
    if cfg.task == "lm":
        ds = PackedLMDataset(cfg.data.train_paths, tokenizer, cfg.data.seq_len, skip_lines=0)
        # packed dataset skip is approximate; we skip sequences via samples_seen below
    else:
        ds = GECDataset(cfg.data.train_paths, tokenizer, cfg.data.seq_len, skip_lines=skip)

    loader = DataLoader(
        ds,
        batch_size=cfg.train.micro_batch_size,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = maybe_compile(model, cfg)
    model.train()
    pad_id = tokenizer.pad_id
    t0 = time.time()
    deadline = t0 + cfg.train.max_hours * 3600
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    tokens = 0
    skipped_samples = 0

    pbar = tqdm(total=cfg.optim.max_steps, initial=start_step, desc=cfg.task)
    micro = 0
    for batch in loader:
        if time.time() >= deadline or step >= cfg.optim.max_steps:
            break
        if cfg.task == "lm" and skipped_samples < samples_seen:
            skipped_samples += batch["token_ids"].size(0)
            continue

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        ctx = torch.autocast(
            device_type=device.type,
            dtype=dtype or torch.float32,
            enabled=dtype is not None and device.type == "cuda",
        )
        with ctx:
            if cfg.task == "lm":
                loss = model.lm_loss(batch["token_ids"], batch["char_ids"], pad_id)
            else:
                loss = model.gec_loss(
                    batch["token_ids"],
                    batch["char_ids"],
                    batch["edit_labels"],
                    pad_id,
                )
            loss = loss / cfg.train.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += float(loss.detach()) * cfg.train.grad_accum
        tokens += int(batch["token_ids"].numel())
        samples_seen += int(batch["token_ids"].size(0))
        micro += 1

        if micro % cfg.train.grad_accum != 0:
            continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        pbar.update(1)

        if step % cfg.train.log_every == 0:
            dt = max(time.time() - t0, 1e-6)
            pbar.set_postfix(loss=f"{running / cfg.train.log_every:.4f}", tok_s=f"{tokens / dt:.0f}")
            running = 0.0
            tokens = 0
            t0 = time.time()

        if step % cfg.train.save_every_steps == 0:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(
                root=root,
                step=step,
                samples_seen=samples_seen,
                model=raw,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                model_cfg=cfg.model,
                extra={"task": cfg.task},
                keep_last=cfg.train.keep_last,
            )

    pbar.close()

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    out = save_checkpoint(
        root=root,
        step=step,
        samples_seen=samples_seen,
        model=raw,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler if scaler.is_enabled() else None,
        model_cfg=cfg.model,
        extra={"task": cfg.task, "stopped_reason": "budget_or_max_steps"},
        keep_last=cfg.train.keep_last,
    )
    (root / "last_run.json").write_text(
        json.dumps({"step": step, "samples_seen": samples_seen, "path": str(out)}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] saved {out}")
    return out
