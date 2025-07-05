import torch
import torch.nn as nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Rotary Positional Embedding (RoPE) for Transformer models.
        Args:
            theta (float): The base frequency for the rotary embeddings.
            d_k (int): The dimension of the key/query vectors.
            max_seq_len (int): The maximum sequence length for which to precompute embeddings.
            device (torch.device | None): The device on which to store the embeddings. Defaults to None.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        base = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d_k, 2, device=device)[: (self.d_k // 2)].float() / self.d_k)
        )
        seq_idx = torch.arange(0, self.max_seq_len, dtype=base.dtype, device=device)
        idx_base = einsum(seq_idx, base, "max_seq_len, d_k_2 -> max_seq_len d_k_2")
        cache = rearrange(
            [torch.cos(idx_base), torch.sin(idx_base)],
            "d_2 max_seq_len d_k_2 -> max_seq_len d_k_2 d_2",
        )

        self.register_buffer(
            "cache",
            cache,
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32
        rope_cache = self.cache[token_positions]
        xshaped = rearrange(
            x,
            "... (d_k_2 d_2) -> ... d_k_2 d_2",
            d_2=2,
        )
        return rearrange(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            "d_2 ... d_k_2 -> ... (d_k_2 d_2)",
        )
