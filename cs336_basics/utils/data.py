import os

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    x: npt.NDArray[np.int64], batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, " batch_size context_length"], Int[Tensor, " batch_size context_length"]]:
    """
    Loads a batch of data from the input array.

    Args:
        x (npt.NDArray): Input data array (integer array with token IDs).
        batch_size (int): Size of the batch.
        context_length (int): Length of the context.
        device (str): Device to load the data onto.

    Returns:
        tuple: A tuple containing two tensors:
            - `inputs`: Indices of the input data.
            - `targets`: Indices of the output data.
    """
    assert context_length > 0, "Context length must be greater than 0."
    assert batch_size > 0, "Batch size must be greater than 0."
    starts = np.random.randint(0, x.shape[0] - context_length, size=batch_size)
    inputs = np.array([x[start : start + context_length] for start in starts], dtype=np.int64)
    targets = np.array(
        [x[start + 1 : start + context_length + 1] for start in starts], dtype=np.int64
    )

    inputs = torch.from_numpy(inputs).to(device)
    targets = torch.from_numpy(targets).to(device)
    return inputs, targets


def load_dataset(
    data_path: str | os.PathLike,
) -> npt.NDArray[np.int64]:
    return np.memmap(data_path, dtype=np.int64, mode="r")
