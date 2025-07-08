import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from cs336_basics.tokenizer import BPETokenizer, find_chunk_boundaries


def _encode(
    start: int,
    end: int,
    input_path: str | os.PathLike,
    pretrained_filepath: str | os.PathLike,
    special_tokens: list[str],
) -> npt.NDArray[np.int64]:
    tokenizer = BPETokenizer.from_files(
        pretrained_filepath=pretrained_filepath,
        special_tokens=special_tokens,
    )
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return np.array(tokenizer.encode(chunk), dtype=np.int64)


def _encode_star(
    args: tuple[int, int, str | os.PathLike, str | os.PathLike, list[str]],
) -> npt.NDArray[np.int64]:
    """Helper function to unpack arguments for multiprocessing."""
    return _encode(*args)


def tokenize_text_file(
    data_path: str | os.PathLike,
    pretrained_filepath: str | os.PathLike,
    special_tokens: list[str] = ["<|endoftext|>"],
    output_path: str | os.PathLike | None = None,
) -> npt.NDArray[np.int64]:
    """
    Tokenizes the input data using the provided tokenizer.
    """
    print(
        f"Tokenizing {data_path} with special tokens {special_tokens}, "
        f"using tokenizer from {pretrained_filepath}. output_path={output_path}"
    )

    start_time = time.time()
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f, split_special_token=b"<|endoftext|>", desired_chunk_size_in_bytes=1024 * 64
        )
    # num_processes = max(1, os.cpu_count() - 4)
    num_processes = os.cpu_count()
    with Pool(processes=num_processes) as pool:
        args = [
            (start, end, data_path, pretrained_filepath, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        print(
            f"Using {num_processes} processes for parallelizing tokenizing of {len(args)} chunks at {data_path}."
        )
        imap_iter = tqdm(pool.imap(_encode_star, args), total=len(args), desc="Tokenizing")
        tokens = np.fromiter(itertools.chain.from_iterable(imap_iter), dtype=np.int64)
    end_time = time.time()
    print(f"Tokenizing took {end_time - start_time:.2f} seconds.")

    print(f"Tokenized {len(tokens)} tokens from {data_path}")

    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(output_path, tokens)
        print(f"Tokens saved to {output_path}.")

    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize a text file using BPE tokenizer.")
    parser.add_argument("--data_path", type=str, help="Path to the input text file.")
    parser.add_argument(
        "--pretrained_filepath", type=str, help="Path to the pretrained BPE tokenizer files."
    )
    parser.add_argument(
        "--special_tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="List of special tokens to include in the tokenizer.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to save the tokenized output."
    )

    args = parser.parse_args()
    tokenize_text_file(
        args.data_path, args.pretrained_filepath, args.special_tokens, args.output_path
    )
