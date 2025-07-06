import torch
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
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


def gradient_clipping_(
    parameters: list[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clips gradients of the parameters to prevent exploding gradients.

    Args:
        parameters (list[torch.nn.Parameter]): List of model parameters.
        max_l2_norm (float): Maximum l2 norm for clipping.
    """
    """
    Computes the total norm of the gradients and clips them if they exceed max_l2_norm.
    """
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters if p.grad is not None]),
        2,
    )
    if total_norm >= max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
