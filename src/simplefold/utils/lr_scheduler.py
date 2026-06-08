#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        **kwargs
    ):
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size

        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(LinearWarmup, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]
        else:
            return [self.max_lr for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.step_in_cycle = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup from min_lr to max_lr, then cosine decay to eta_min."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        T_max: int = 100000,
        eta_min: float = None,
        last_epoch: int = -1,
        **kwargs
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if T_max <= 0:
            raise ValueError("T_max must be > 0")

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = min_lr if eta_min is None else eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def _compute_lr(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            pct = step / self.warmup_steps
            return self.min_lr + (self.max_lr - self.min_lr) * pct

        decay_step = max(0, step - self.warmup_steps)
        pct = min(decay_step / self.T_max, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * pct))
        return self.eta_min + (self.max_lr - self.eta_min) * cosine

    def get_lr(self):
        step = max(self.last_epoch, 0)
        return [self._compute_lr(step) for _ in self.optimizer.param_groups]
