"""Atomic, Windows-safe checkpoints. Leave a machine for 1–2 days and resume."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from typotaker.config import ModelConfig

SCHEMA = 1
LATEST = "latest.json"


def run_dir(checkpoint_dir: str | Path, run_name: str) -> Path:
    return Path(checkpoint_dir) / run_name


def _step_dir(root: Path, step: int) -> Path:
    return root / f"step_{step:08d}"


def save_checkpoint(
    *,
    root: Path,
    step: int,
    samples_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    model_cfg: ModelConfig,
    extra: dict[str, Any] | None = None,
    keep_last: int = 4,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = _step_dir(root, step)
    tmp = root / f".tmp_step_{step:08d}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, tmp / "model.pt")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), tmp / "scaler.pt")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "rng.pt",
    )
    meta = {
        "schema": SCHEMA,
        "step": step,
        "samples_seen": samples_seen,
        "model": asdict(model_cfg),
        **(extra or {}),
    }
    (tmp / "trainer_state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _write_latest(root, step, final)
    _prune(root, keep_last)
    return final


def _write_latest(root: Path, step: int, path: Path) -> None:
    payload = {"step": step, "path": str(path.as_posix())}
    tmp = root / ".latest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, root / LATEST)


def _prune(root: Path, keep_last: int) -> None:
    steps = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: p.name,
    )
    for old in steps[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


def find_latest(root: Path) -> Path | None:
    latest = root / LATEST
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        path = Path(data["path"])
        if path.exists():
            return path
    steps = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: p.name,
    )
    return steps[-1] if steps else None


def load_checkpoint(path: Path) -> dict[str, Any]:
    meta = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    blob = {
        "meta": meta,
        "model": torch.load(path / "model.pt", map_location="cpu", weights_only=True),
        "optimizer": torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False),
        "rng": torch.load(path / "rng.pt", map_location="cpu", weights_only=False),
    }
    sched = path / "scheduler.pt"
    scaler = path / "scaler.pt"
    if sched.exists():
        blob["scheduler"] = torch.load(sched, map_location="cpu", weights_only=True)
    if scaler.exists():
        blob["scaler"] = torch.load(scaler, map_location="cpu", weights_only=True)
    return blob


def restore_rng(blob: dict[str, Any]) -> None:
    rng = blob.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass
