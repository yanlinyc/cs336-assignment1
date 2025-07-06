from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    input: Float[Tensor, "... batch_size seq_len vocab_size"],
    target: Int[Tensor, "... batch_size seq_len"],
) -> Float[Tensor, ""]:
    shifted = input - input.amax(dim=-1, keepdim=True)  # for numerical stability
    logz: Float[Tensor, "... batch_size seq_len"] = shifted.exp().sum(dim=-1).log()
    gathered: Float[Tensor, "... batch_size seq_len"] = shifted.gather(
        dim=-1, index=target.unsqueeze(-1)
    ).squeeze(-1)

    loss = logz - gathered
    return loss.mean()
