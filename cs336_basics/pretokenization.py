import os
from typing import BinaryIO


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
