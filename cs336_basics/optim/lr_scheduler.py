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
