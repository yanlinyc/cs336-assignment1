import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Int
from torch import Tensor

from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import BPETokenizer


def generate_text(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    input: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(input)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    input_ids = rearrange(input_ids, "seq_len -> 1 seq_len")  # Add batch dimension

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits: Int[Tensor, " 1 seq_len vocab_size"] = model(input_ids)
        next_token_logits: Int[Tensor, " 1 vocab_size"] = logits[:, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        probs = rearrange(F.softmax(next_token_logits, dim=-1), " 1 vocab_size -> vocab_size")
        probs = top_p_sampling(probs, top_p)

        next_token = torch.multinomial(probs, num_samples=1).item()

        # Append to sequence
        input_ids: Int[Tensor, " 1 seq_len"] = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=model.device)], dim=1
        )

        if next_token == tokenizer.eos_token_id:
            break

    generated_ids = input_ids[0, input_len:].tolist()
    return tokenizer.decode(generated_ids)


def top_p_sampling(probs, top_p):
    """Filter a distribution of logits using nucleus (top-p) sampling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative prob > top_p
    sorted_mask = cumulative_probs > top_p
    # Shift mask one to the right to always keep at least one token
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0

    # Set probs of filtered tokens to 0
    probs[sorted_indices[sorted_mask]] = 0
    probs = probs / probs.sum()  # renormalize
    return probs
