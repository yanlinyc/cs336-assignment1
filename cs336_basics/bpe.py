import regex as re
from collections import Counter

type TokenCounter = dict[tuple[bytes], int]


def merge_tokens(cur_tokens: TokenCounter, pair_counts: TokenCounter) -> tuple[TokenCounter, bytes]:
    # Find the most frequent pair
    best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
    new_token = best_pair[0] + best_pair[1]

    def dec_(pair: tuple[bytes], count: int):
        pair_counts[pair] -= count
        if pair_counts[pair] == 0:
            del pair_counts[pair]

    new_tokens: TokenCounter = Counter()
    for tokens, count in cur_tokens.items():
        # Replace occurrences of best_pair with new_token
        new_sequence: list[bytes] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_sequence.append(new_token)

                # Update pair counts for adjacent tokens

                if i > 0 and (prev_token := tokens[i - 1],):
                    pair_counts[(prev_token, new_token)] += count
                    dec_((prev_token, best_pair[0]), count)

                if i + 2 < len(tokens) and (next_token := tokens[i + 2],):
                    pair_counts[(new_token, next_token)] += count
                    dec_((best_pair[1], next_token), count)

                i += 2
            else:
                new_sequence.append(tokens[i])
                i += 1
        new_tokens[tuple(new_sequence)] += count

    del pair_counts[best_pair]
    return new_tokens, new_token


def train_bpe_tokenizer(
    text: str,
    num_merges: int = 1000,
    special_tokens: list[str] = ["<|endoftext|>"],
    debug: bool = False,
):
    vocab = [s.encode("utf-8") for s in special_tokens] + [k.to_bytes() for k in range(256)]

    # pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens: TokenCounter = Counter()
    for match in re.finditer(PAT, text):
        pre_tokens[tuple(b.to_bytes() for b in match.group(0).encode("utf-8"))] += 1

    pair_counts: TokenCounter = Counter()
    for tokens, count in pre_tokens.items():
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += count

    cur_tokens = pre_tokens
    for iter in range(num_merges):
        if not cur_tokens:
            print("[WARN] No more pairs to merge.")
            break
        if debug:
            print("---")
            print(f"Current pairs:")
            print(pair_counts)

        cur_tokens, new_token = merge_tokens(cur_tokens, pair_counts)
        vocab.append(new_token)

        if debug:
            print(f"Merge {iter + 1}/{num_merges}: {new_token.decode('utf-8')}")
    return vocab


if __name__ == "__main__":
    test_texts = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
    """.strip()

    vocab = train_bpe_tokenizer(test_texts, num_merges=6, debug=True)
    print("Vocabulary size:", len(vocab))
    print("Sample vocabulary:", vocab[257:])
