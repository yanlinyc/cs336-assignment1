import os
from pathlib import Path
from multiprocessing import Pool
import regex as re
from collections import Counter, defaultdict
import time

from tqdm.auto import tqdm

from cs336_basics.pretokenization import find_chunk_boundaries
from cs336_basics.io_utils import save_pickle, load_pickle

type TokenCounter = dict[tuple[bytes], int]


def _apply_pair_updates(
    pair_counts: TokenCounter, pair_updates: dict[tuple[bytes, bytes], int]
) -> None:
    """Apply batched updates to pair_counts."""
    for pair, delta in pair_updates.items():
        if delta != 0:
            new_count = pair_counts.get(pair, 0) + delta
            if new_count > 0:
                pair_counts[pair] = new_count
            elif pair in pair_counts:
                del pair_counts[pair]


def merge_tokens(
    cur_tokens: TokenCounter,
    pair_counts: TokenCounter,
) -> tuple[TokenCounter, bytes, tuple[bytes, bytes]]:

    best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
    new_token = best_pair[0] + best_pair[1]

    # Create lookup for O(1) membership testing
    target_first, target_second = best_pair

    new_tokens: TokenCounter = Counter()
    pair_updates = defaultdict(int)

    for tokens, count in cur_tokens.items():
        # Skip short sequences that can't contain pairs
        if len(tokens) < 2:
            new_tokens[tokens] = count
            continue

        # Fast scan for target pair existence
        has_target = False
        for i in range(len(tokens) - 1):
            if tokens[i] == target_first and tokens[i + 1] == target_second:
                has_target = True
                break

        if not has_target:
            new_tokens[tokens] = count
            continue

        # Process sequence with target pair
        new_sequence = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == target_first and tokens[i + 1] == target_second:

                # Record pair changes around merge point
                if i > 0:
                    old_pair = (tokens[i - 1], target_first)
                    new_pair = (tokens[i - 1], new_token)
                    pair_updates[old_pair] -= count
                    pair_updates[new_pair] += count

                if i + 2 < len(tokens):
                    old_pair = (target_second, tokens[i + 2])
                    new_pair = (new_token, tokens[i + 2])
                    pair_updates[old_pair] -= count
                    pair_updates[new_pair] += count

                new_sequence.append(new_token)
                i += 2
            else:
                new_sequence.append(tokens[i])
                i += 1

        new_tokens[tuple(new_sequence)] += count

    # Apply updates
    _apply_pair_updates(pair_counts, pair_updates)
    del pair_counts[best_pair]

    return new_tokens, new_token, best_pair


def pre_tokenize_impl(
    start: int, end: int, input_path: str | os.PathLike, special_tokens: list[str]
) -> TokenCounter:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pattern = "|".join(re.escape(token) for token in special_tokens)
    splits = re.split(pattern, chunk)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens: TokenCounter = Counter()
    for text in splits:
        for match in re.finditer(PAT, text):
            pre_tokens[tuple(b.to_bytes() for b in match.group(0).encode("utf-8"))] += 1

    return pre_tokens


def pre_tokenize_star(args: tuple[int, int, str | os.PathLike, list[str]]) -> TokenCounter:
    """Helper function to unpack arguments for multiprocessing."""
    return pre_tokenize_impl(*args)


def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    output_dir: str | os.PathLike | None,
) -> TokenCounter:
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
        pre_tokens: TokenCounter = Counter()
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


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"],
    output_dir: str | os.PathLike | None = None,
    pre_tokens_path: str | os.PathLike | None = None,
    logging_steps: int = 1000,
    debug: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    print(
        f"Training BPE tokenizer on {input_path} with vocab size {vocab_size}, special tokens {special_tokens}..."
    )

    if "<|endoftext|>" not in special_tokens:
        raise ValueError("At least one special token <|endoftext|> must be provided.")
    vocab = [s.encode("utf-8") for s in special_tokens] + [k.to_bytes() for k in range(256)]

    if vocab_size <= len(vocab):
        raise ValueError(
            f"Vocabulary size {vocab_size} must be greater than the number of special tokens {len(vocab)}."
        )
    if pre_tokens_path:
        print(f"Loading pre-tokens from {pre_tokens_path}...")
        pre_tokens = load_pickle(pre_tokens_path)
        print(f"Loaded {len(pre_tokens)} pre-tokens.")
    else:
        print("Pre-tokenizing input data...")
        pre_tokens = pre_tokenize(
            input_path=input_path,
            special_tokens=special_tokens,
            output_dir=output_dir,
        )

    if debug:
        u_pre_tokens = sorted(set((b"".join(k)).decode("utf-8") for k in pre_tokens.keys()))
        print(f"Unique pre-tokens: {len(u_pre_tokens)}")
        print(f"First 10 unique pre-tokens: {u_pre_tokens[:10]}")
        print(f"Last 10 unique pre-tokens: {u_pre_tokens[-10:]}")
        print(
            f"Random 10 unique pre-tokens: {u_pre_tokens[::max(1, len(u_pre_tokens) // 10)][:10]}"
        )

    num_merges = vocab_size - len(vocab)
    print(f"Starting BPE training with {num_merges} merges.")

    pair_counts: TokenCounter = Counter()
    for tokens, count in pre_tokens.items():
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += count

    cur_tokens = pre_tokens
    merges = []
    for iter in tqdm(range(num_merges), desc="Training BPE"):
        if not cur_tokens:
            print("[WARN] No more pairs to merge.")
            break
        if debug:
            print("---")
            print(f"Current pairs:")
            print(pair_counts)

        cur_tokens, new_token, best_pair = merge_tokens(cur_tokens, pair_counts)
        vocab.append(new_token)
        merges.append(best_pair)

        if (iter + 1) % logging_steps == 0:
            print(
                f"- Iteration {iter + 1}/{num_merges}: {len(cur_tokens)} tokens, {len(pair_counts)} pairs, "
                f"Merged `{new_token.decode('utf-8')}`"
            )

        if debug:
            print(f"Merge {iter + 1}/{num_merges}: {new_token.decode('utf-8')}")

    result = ({i: v for i, v in enumerate(vocab)}, merges)
    if output_dir:
        f_name = os.path.join(output_dir, f"{Path(input_path).stem}-bpe.pkl")
        print(f"Saving vocabulary and merges to {f_name}...")
        save_pickle(f_name, result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "--profile",
        type=bool,
        default=False,
        help="Enable profiling to measure performance.",
    )
    parser.add_argument("--input_path", type=str, help="Path to the input corpus.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/output",
        help="Directory to save the trained tokenizer vocabulary and merges.",
    )
    parser.add_argument(
        "--pre_tokens_path",
        type=str,
        default=None,
        help="Path to the pre-tokenized input data.",
    )
    parser.add_argument("--vocab_size", type=int, default=10000, help="Total vocabulary size.")
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<|endoftext|>"],
        help="List of special tokens to be added to the vocabulary.",
    )
    args = parser.parse_args()

    if args.profile:
        from scalene import scalene_profiler

        scalene_profiler.start()

    vocab, merges = train_bpe_tokenizer(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        output_dir=args.output_dir,
        pre_tokens_path=args.pre_tokens_path,
    )
    if args.profile:
        scalene_profiler.stop()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print("First 10 merges:", merges[:10])
