import math
import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce


def scale_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k).
        key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k).
        value (torch.Tensor): Value tensor of shape (..., seq_len_k, d_v).
        mask (torch.Tensor | None): Optional mask tensor of shape (..., seq_len_q, seq_len_k).

    Returns:
        torch.Tensor: Output tensor of shape (..., seq_len_q, d_v).
    """
    d_k = query.shape[-1]
    scores = einsum(
        query, key, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    ) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return einsum(
        attn_weights, value, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v"
    )


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
