import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the softmax of the input tensor along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_values)  # for numerical stability
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
