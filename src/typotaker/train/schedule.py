from __future__ import annotations

import math

import torch


def cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, max_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
