import os
from pathlib import Path
from collections import defaultdict
import heapq

from tqdm.auto import tqdm
from sortedcontainers import SortedSet

from cs336_basics.pretokenization import pre_tokenize
from cs336_basics.io_utils import save_pickle, load_pickle


def _sanity_check_occurrences(
    occurrences: dict[tuple[bytes, bytes], set[tuple[int, int]]],
    tokens_sequence: list[list[bytes]],
):
    """Check that occurrences are consistent with tokens_sequence."""
    expected_occurrences = defaultdict(set)
    for seq_id, tokens in enumerate(tokens_sequence):
        for pos in range(len(tokens) - 1):
            pair = (tokens[pos], tokens[pos + 1])
            expected_occurrences[pair].add((seq_id, pos))

    for pair, occ in occurrences.items():
        if occ != expected_occurrences[pair]:
            raise ValueError(
                f"Inconsistent occurrences for pair {pair}: "
                f"expected {expected_occurrences[pair]}, got {occ}"
            )


def train_bpe_tokenizer_optim(
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
        pre_tokens: dict[tuple[bytes, ...], int] = load_pickle(pre_tokens_path)
        print(f"Loaded {len(pre_tokens)} pre-tokens.")
    else:
        print("Pre-tokenizing input data...")
        pre_tokens: dict[tuple[bytes, ...], int] = pre_tokenize(
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

    # pq: list[tuple[int, tuple[bytes, bytes]]] = []
    tokens_sequence: list[list[bytes]] = [list(k) for k in pre_tokens]
    tokens_counts: list[int] = list(pre_tokens[tuple(k)] for k in tokens_sequence)
    occurrences: dict[tuple[bytes, bytes], dict[int, SortedSet[int]]] = defaultdict(
        lambda: defaultdict(SortedSet)
    )
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for seq_id, (tokens, count) in enumerate(pre_tokens.items()):
        for pos in range(len(tokens) - 1):
            pair_counts[(tokens[pos], tokens[pos + 1])] += count
            occurrences[(tokens[pos], tokens[pos + 1])][seq_id].add(pos)

    #     print(f"Initial pre_tokens:")
    #     print(pre_tokens)
    #     print(f"Initial tokens_sequence:")
    #     print(tokens_sequence)
    #     print(f"Initial tokens_counts:")
    #     print(tokens_counts)
    #     print(f"Initial occurrences:")
    #     print(occurrences)

    def _discard(pair: tuple[bytes, bytes], seq: int, pos: int):
        occurrences[pair][seq].discard(pos)
        if not occurrences[pair][seq]:
            occurrences[pair].pop(seq, None)
        if not occurrences[pair]:
            occurrences.pop(pair, None)

    merges = []
    result = ({i: v for i, v in enumerate(vocab)}, merges)
    for iter in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:
            print("No more pairs to merge. Stopping early.")
            break

        # Get the most frequent pair
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        a, b = best_pair
        new_token = a + b
        pair_updates: dict[tuple[bytes, bytes], int] = defaultdict(int)
        seq_pos = occurrences[best_pair]

        if debug:
            print("---")
            print(f"Merge {iter + 1}/{num_merges}: {best_pair}")
            print(f"Current pairs:")
            print(sorted(pair_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[:10])
            print('occurrences: (b"h", b"e")')
            print(occurrences[(b"h", b"e")])
            # print(f"current occurrences:")
            # print(occurrences)
            # print(f"tokens_sequence:")
            # print(tokens_sequence)
            # print(f"tokens_counts:")
            # print(tokens_counts)

        debug_p = iter >= 2
        if debug_p:

            print("sequence positions for best pair:", seq_pos)
            for seq_id, pos in seq_pos.items():
                print(
                    f"seq_id: {seq_id}, pos: {pos}, tokens: {tokens_sequence[seq_id]}, count: {tokens_counts[seq_id]}"
                )
        sequence_ids = list(seq_pos.keys())

        for seq_id in sequence_ids:
            if seq_id not in seq_pos:
                continue
            positions = seq_pos[seq_id]

            cur_tokens = tokens_sequence[seq_id]
            count = tokens_counts[seq_id]
            while positions:
                pos = positions.pop(0)
                # x (pos-1), a (pos), b (pos+1), y (pos+2) -> x (pos-1), new_token (pos), y (pos+1)
                if pos > 0:
                    old_pair = (cur_tokens[pos - 1], a)
                    new_pair = (cur_tokens[pos - 1], new_token)
                    pair_updates[old_pair] -= count
                    pair_updates[new_pair] += count
                    _discard(old_pair, seq_id, pos - 1)
                    occurrences[new_pair][seq_id].add(pos - 1)
                    if debug_p:
                        print("old_pair:", old_pair, "new_pair:", new_pair, "count:", count)
                        print(pair_updates[old_pair], pair_updates[new_pair])

                if pos + 2 < len(cur_tokens):
                    old_pair = (b, cur_tokens[pos + 2])
                    new_pair = (new_token, cur_tokens[pos + 2])
                    pair_updates[old_pair] -= count
                    pair_updates[new_pair] += count
                    _discard(old_pair, seq_id, pos + 1)
                    occurrences[new_pair][seq_id].add(pos)
                    if debug_p:
                        print("old_pair:", old_pair, "new_pair:", new_pair, "count:", count)
                        print(pair_updates[old_pair], pair_updates[new_pair])

                for i in range(pos + 2, len(cur_tokens) - 1):
                    pair = (cur_tokens[i], cur_tokens[i + 1])
                    _discard(pair, seq_id, i)
                    occurrences[pair][seq_id].add(i - 1)

                cur_tokens[pos : pos + 2] = [new_token]

        if debug_p:
            print("pair_updates:")
            print(pair_updates)

        # Update pair counts
        for pair, delta in pair_updates.items():
            if delta != 0:
                new_count = pair_counts.get(pair, 0) + delta
                if new_count > 0:
                    pair_counts[pair] = new_count
                else:
                    pair_counts.pop(pair, None)

        occurrences.pop(best_pair, None)
        pair_counts.pop(best_pair, None)

        vocab.append(new_token)
        merges.append(best_pair)

        if (iter + 1) % logging_steps == 0:
            print(
                f"- Iteration {iter + 1}/{num_merges}: {len(cur_tokens)} tokens, {len(pair_counts)} pairs, "
                f"Merged `{new_token.decode('utf-8')}`"
            )

            _sanity_check_occurrences(occurrences, tokens_sequence)

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
