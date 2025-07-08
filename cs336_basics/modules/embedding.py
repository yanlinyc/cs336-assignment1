import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Embedding layer that initializes weights with a truncated normal distribution.
        Args:
            num_embeddings (int): Number of unique tokens in the vocabulary.
            embedding_dim (int): Dimensionality of the embeddings, d_model
            device (torch.device | None): Device to place the embeddings on. Defaults to None.
            dtype (torch.dtype | None): Data type of the embeddings. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]
