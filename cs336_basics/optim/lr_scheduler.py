import math

import torch


def get_lr_cosine_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Returns the learning rate for the current iteration using a cosine schedule with warmup.

    Args:
        it (int): Current iteration number.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum (final) learning rate.
        warmup_iters (int): Number of warmup iterations.
        cosine_cycle_iters (int): Number  of cosine annealing iterations.

    Returns:
        float: Learning rate for the current iteration.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_lr

    if it <= cosine_cycle_iters:
        cos_inner = torch.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_lr + 0.5 * (max_lr - min_lr) * (math.cos(cos_inner) + 1)

    return min_lr


class LRScheduler:
    def __call__(self, it: int) -> float:
        raise NotImplementedError(
            "LRScheduler is an abstract class, please implement the __call__ method."
        )


class CosineLRScheduler(LRScheduler):
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it: int) -> float:
        return get_lr_cosine_schedule(
            it, self.max_lr, self.min_lr, self.warmup_iters, self.cosine_cycle_iters
        )


class ConstantLRScheduler(LRScheduler):
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, it: int) -> float:
        return self.lr
