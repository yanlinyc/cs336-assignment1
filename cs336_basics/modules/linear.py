import math
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Linear layer that initializes weights with a truncated normal distribution.
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (torch.device | None): Device to place the weights on. Defaults to None.
            dtype (torch.dtype | None): Data type of the weights. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(
            self.weight,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
