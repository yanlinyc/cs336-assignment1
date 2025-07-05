import torch
import torch.nn as nn
from einops import einsum, reduce


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        RMSNorm layer that normalizes the input tensor along the last dimension.
        Args:
            d_model (int): Dimensionality of the input tensor.
            eps (float): Small value to avoid division by zero.
            device (torch.device | None): Device to place the weights on. Defaults to None.
            dtype (torch.dtype | None): Data type of the weights. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # Convert to float32 for numerical stability to avoid overflow when squaring
        x = x.to(torch.float32)
        norm = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        result = einsum(x / norm, self.weight, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)
