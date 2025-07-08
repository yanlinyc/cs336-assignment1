import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current iteration number.
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The output file path or file-like object.
    """
    to_save = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if hasattr(model, "config"):
        to_save["model_config"] = model.config
    if hasattr(optimizer, "config"):
        to_save["optimizer_config"] = optimizer.config

    torch.save(to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The source file path or file-like object.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def load_from_pretrained(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    from cs336_basics.modules import TransformerLM
    from cs336_basics.optim import AdamW

    checkpoint = torch.load(src)
    model = TransformerLM.from_pretrained(
        config=checkpoint["model_config"], state_dict=checkpoint["model_state_dict"]
    )
    optimizer = AdamW.from_pretrained(
        model, config=checkpoint["optimizer_config"], state_dict=checkpoint["optimizer_state_dict"]
    )
    iteration = checkpoint["iteration"]
    return model, optimizer, iteration
