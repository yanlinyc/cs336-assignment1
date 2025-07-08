from collections.abc import Callable
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


def run_sgd(lr: float, weights: torch.nn.Parameter, num_iterations: int = 100) -> list[float]:
    print(f"Running SGD with learning rate: {lr} for {num_iterations} iterations...")

    opt = SGD([weights], lr=lr)
    losses = []  # List to store loss values for each iteration.
    for _ in range(num_iterations):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
        losses.append(loss.cpu().item())
    return losses
