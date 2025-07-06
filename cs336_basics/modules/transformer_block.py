import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int


from .rope import RotaryPositionalEmbedding
from .norm import RMSNorm
from .ffn import SwiGLU
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope=rope,
            **factory_kwargs,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        x_normed = self.ln1(x)
        attn_output = self.attn(x_normed)
        y = x + attn_output
        y_normed = self.ln2(y)
        ffn_output = self.ffn(y_normed)
        return y + ffn_output
