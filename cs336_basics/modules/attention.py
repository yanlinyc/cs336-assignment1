import math
import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int

from cs336_basics.functional import softmax

from .linear import Linear
from .rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Multi-head self-attention layer.
        Args:
            d_model (int): Dimensionality of the input and output vectors.
            num_heads (int): Number of attention heads.
            max_seq_len (int | None): Maximum sequence length for which to create a causal mask. Defaults to None.
            rope (RotaryPositionalEmbedding | None): Rotary positional embedding instance. Defaults to None.
            device (torch.device | None): Device to place the weights on. Defaults to None.
            dtype (torch.dtype | None): Data type of the weights. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rope = rope

        # Combined QKV projection for efficiency
        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        if max_seq_len is not None:
            mask = self._create_causal_mask(max_seq_len, device=device)
            self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        # x: (..., seq_len, d_model)
        seq_len = x.shape[-2]
        if self.max_seq_len is not None:
            mask = self.causal_mask[:seq_len, :seq_len]
        else:
            mask = self._create_causal_mask(seq_len, device=x.device)

        qkv = self.qkv_proj(x)  # (..., seq_len, 3 * d_model)
        query, key, value = rearrange(
            qkv,
            "... seq_len (stack num_heads d_k) -> stack ... num_heads seq_len d_k",
            stack=3,
            num_heads=self.num_heads,
        )

        # apply shared RoPE if provided
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        attn_output = scaled_dot_product_attention(query, key, value, mask=mask)
        attn_output = rearrange(
            attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )
        return self.o_proj(attn_output)

    def _create_causal_mask(
        self, seq_len: int, device: torch.device | None = None
    ) -> Bool[Tensor, "seq_len seq_len"]:
        """
        Creates a causal mask for the attention mechanism.

        Args:
            seq_len (int): The sequence length for which to create the mask.

        Returns:
            torch.Tensor: A causal mask of shape (seq_len, seq_len).
        """
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()


def scaled_dot_product_attention(
    query: Float[Tensor, " ... seq_len_q d_k"],
    key: Float[Tensor, " ... seq_len_k d_k"],
    value: Float[Tensor, " ... seq_len_k d_v"],
    mask: Bool[Tensor, " ... seq_len_q seq_len_k"] | None = None,
) -> Float[Tensor, " ... seq_len_q d_v"]:
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
