# scheduler.py

from __future__ import annotations

import math
from enum import Enum
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class SchedulerType(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"


class WarmupScheduler:
    """
    Фабрика LR-шедулеров с линейным warmup.

    Пример:
        scheduler = WarmupScheduler.create(
            optimizer,
            name="cosine",
            num_warmup_steps=100,
            num_training_steps=1000,
            min_lr=1e-5,
        )
    """

    @staticmethod
    def create(
        optimizer: Optimizer,
        name: str | SchedulerType,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> LambdaLR:
        try:
            scheduler_type = SchedulerType(name)
        except ValueError:
            supported = [e.value for e in SchedulerType]
            raise ValueError(f"Unknown scheduler '{name}'. Supported: {supported}")

        # Вычисляем min_lr_ratio относительно каждой группы параметров
        base_lrs = [group["lr"] for group in optimizer.param_groups]

        builders = {
            SchedulerType.COSINE: WarmupScheduler._cosine_lambda,
            SchedulerType.LINEAR: WarmupScheduler._linear_lambda,
        }
        builder = builders[scheduler_type]

        lr_lambdas = [
            builder(
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=min_lr / base_lr if base_lr > 0 else 0.0,
            )
            for base_lr in base_lrs
        ]

        return LambdaLR(optimizer, lr_lambda=lr_lambdas, last_epoch=last_epoch)

    # ── Warmup ───────────────────────────────────────────────────────────

    @staticmethod
    def _warmup_factor(step: int, num_warmup_steps: int) -> float:
        if num_warmup_steps == 0:
            return 1.0
        return min(1.0, step / num_warmup_steps)

    # ── Lambda factories ─────────────────────────────────────────────────

    @staticmethod
    def _cosine_lambda(num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float):
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return WarmupScheduler._warmup_factor(step, num_warmup_steps)
            progress = min(1.0, (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return lr_lambda

    @staticmethod
    def _linear_lambda(num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float):
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return WarmupScheduler._warmup_factor(step, num_warmup_steps)
            progress = min(1.0, (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        return lr_lambda