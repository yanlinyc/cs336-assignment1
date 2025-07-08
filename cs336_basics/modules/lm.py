import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .embedding import Embedding
from .linear import Linear
from .norm import RMSNorm
from .rope import RotaryPositionalEmbedding
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "device": device,
            "dtype": dtype,
        }
        self.canonical_name = (
            f"TransformerLM-vocab_{vocab_size}-context_{context_length}-layers_{num_layers}"
            f"-d_model_{d_model}-num_heads_{num_heads}-d_ff_{d_ff}-rope_theta_{rope_theta}"
        )

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs
        )
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope=self.rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self,
        x: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_model"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
