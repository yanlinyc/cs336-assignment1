import torch
import torch.nn as nn

from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Position-Wise Feed-Forward Network that applies a SwiGLU activation.
        Args:
            d_model (int): The input and output dimension of the model.
            d_ff (int | None): The hidden dimension of the feed-forward network.
            device (torch.device | None): The device to place the module on.
            dtype (torch.dtype | None): The data type of the module's parameters.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            # Default to 8/3 of d_model, rounded down to nearest multiple of 64
            self.d_ff = (d_model * 8.0 / 3.0) // 64 * 64
        else:
            self.d_ff = d_ff

        self.fc1 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.fc2 = Linear(self.d_ff, self.d_model, **factory_kwargs)
        self.fc3 = Linear(self.d_model, self.d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        silu = torch.sigmoid(x1) * x1  # Swish activation
        x2 = silu * (self.fc3(x))
        return self.fc2(x2)
