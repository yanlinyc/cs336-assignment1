import os
from pathlib import Path
from multiprocessing import Pool
import regex as re
from collections import Counter
import time
from typing import BinaryIO

from tqdm.auto import tqdm
from cs336_basics.io_utils import save_pickle


def pre_tokenize_impl(
    start: int, end: int, input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pattern = "|".join(re.escape(token) for token in special_tokens)
    splits = re.split(pattern, chunk)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens: dict[tuple[bytes, ...], int] = Counter()
    for text in splits:
        for match in re.finditer(PAT, text):
            pre_tokens[tuple(b.to_bytes() for b in match.group(0).encode("utf-8"))] += 1

    return pre_tokens


def pre_tokenize_star(
    args: tuple[int, int, str | os.PathLike, list[str]],
) -> dict[tuple[bytes, ...], int]:
    """Helper function to unpack arguments for multiprocessing."""
    return pre_tokenize_impl(*args)


def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    output_dir: str | os.PathLike | None,
) -> dict[tuple[bytes, ...], int]:
    start_time = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            split_special_token="<|endoftext|>".encode("utf-8"),
            desired_chunk_size_in_bytes=1024 * 1024 * 64,  # 64 MB per chunk
        )

    num_processes = max(1, os.cpu_count() - 4)
    with Pool(processes=num_processes) as pool:
        args = [
            (start, end, input_path, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        print(
            f"Using {num_processes} processes for parallelizing pre-tokenization of {len(args)} chunks."
        )
        pre_tokens_iter = tqdm(
            pool.imap_unordered(pre_tokenize_star, args), total=len(args), desc="Pre-tokenizing"
        )
        pre_tokens: dict[tuple[bytes, ...], int] = Counter()
        for tokens in pre_tokens_iter:
            pre_tokens.update(tokens)
    end_time = time.time()
    print(
        f"Pre-tokenization took {end_time - start_time:.2f} seconds. "
        f"Found {len(pre_tokens)} unique tokens."
    )

    if output_dir:
        output_file = os.path.join(output_dir, f"{Path(input_path).stem}-pre_tokens.pkl")
        save_pickle(output_file, pre_tokens)
        print(f"Pre-tokens saved to {output_file}.")
    return pre_tokens


def find_chunk_boundaries(
    file: BinaryIO,
    split_special_token: bytes,
    desired_num_chunks: int | None = None,
    desired_chunk_size_in_bytes: int | None = None,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    # either desired_num_chunks or desired_chunk_size must be provided
    assert (desired_num_chunks is not None) ^ (
        desired_chunk_size_in_bytes is not None
    ), "Must provide either desired_num_chunks or desired_chunk_size, but not both"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks is not None:
        chunk_size = file_size // desired_num_chunks
        num_chunks = desired_num_chunks
    else:
        chunk_size = desired_chunk_size_in_bytes
        num_chunks = (file_size + chunk_size - 1) // chunk_size

    print(
        f"Chunking file of size {file_size} bytes into {num_chunks} chunks of size {chunk_size} bytes each."
    )

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
