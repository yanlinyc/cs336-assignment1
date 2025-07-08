import math
from collections.abc import Callable
from typing import Self

import torch
from torch.optim.optimizer import ParamsT

from .lr_scheduler import LRScheduler


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        lr_scheduler: LRScheduler | None = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        self.lr_scheduler = lr_scheduler
        self.config = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        if lr_scheduler is not None:
            self.config["lr_scheduler_config"] = lr_scheduler.config
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            t = None
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                grad = p.grad.data
                t = state["t"] + 1
                m = state["m"]
                v = state["v"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                state["t"] = t
                state["m"] = m
                state["v"] = v

            if self.lr_scheduler is not None:
                assert t is not None, "Learning rate scheduler requires the iteration count."
                group["lr"] = self.lr_scheduler(t)

        return loss

    @classmethod
    def from_pretrained(
        cls: type[Self], model: torch.nn.Module, config: dict, state_dict: dict | None = None
    ) -> Self:
        lr_scheduler_config = config.pop("lr_scheduler_config", None)
        lr_scheduler = None
        if lr_scheduler_config:
            lr_scheduler = (LRScheduler.from_pretrained(lr_scheduler_config))
        optimizer = cls(model.parameters(), lr_scheduler=lr_scheduler, **config)
        if state_dict:
            optimizer.load_state_dict(state_dict)
        return optimizer
