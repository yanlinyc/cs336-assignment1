from collections.abc import Iterable, Iterator
from typing import Self

import regex as re

from cs336_basics.utils import load_pickle


class BPETokenizer:
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.merges = merges.copy()
        self.vocab = vocab.copy()
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.token2id = {v: k for k, v in self.vocab.items()}
        self.debug = False
        self.eos_token_id = self.token2id[b"<|endoftext|>"]
        self.merge_ranks = {
            pair: i for i, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls: type[Self],
        pretrained_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Load a BPE tokenizer from files."""
        vocab, merges = load_pickle(pretrained_filepath)
        return cls(vocab, merges, special_tokens=special_tokens)

    def bpe_encode(self, parts: list[bytes]) -> list[bytes]:
        while True:
            pairs = set(zip(parts, parts[1:]))
            if not pairs:
                break

            # Find the merge with the lowest rank (highest priority).
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))

            # If the best pair isn't in our merge rules, we're done.
            if self.merge_ranks.get(best_pair) is None:
                break

            # Merge the best pair.
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> list[int]:
        return self.encode_optim(text)

    def encode_optim(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        pre_tokens = self._pre_tokenize(text)
        if self.debug:
            print(f"[DEBUG] Original text: {text}")
            print(f"[DEBUG] Pre-tokenized text: {pre_tokens}")

        results = []
        for pre_token in pre_tokens:
            if len(pre_token) == 1 and pre_token[0] in self.token2id:
                results.extend(pre_token)
                continue
            results.extend(self.bpe_encode(pre_token))

        if self.debug:
            print(f"[DEBUG] Merged tokens: {results}")
        return [self.token2id[token] for token in results]

    def encode_brute_force(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        pre_tokens = self._pre_tokenize(text)
        if self.debug:
            print(f"[DEBUG] Original text: {text}")
            print(f"[DEBUG] Pre-tokenized text: {pre_tokens}")
        for merge in self.merges:
            for idx, pre_token in enumerate(pre_tokens):
                if len(pre_token) < 2:
                    continue
                i = 0
                while i < len(pre_token) - 1:
                    pair = (pre_token[i], pre_token[i + 1])
                    if pair == merge:
                        pre_token[i : i + 2] = [pre_token[i] + pre_token[i + 1]]
                    else:
                        i += 1
                pre_tokens[idx] = pre_token
        results = sum(pre_tokens, [])
        if self.debug:
            print(f"[DEBUG] Merged tokens: {results}")
        return [self.token2id[token] for token in results]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        from tqdm.auto import tqdm

        if hasattr(iterable, "seek") and callable(iterable.seek):
            total = sum(1 for _ in iterable)
            iterable.seek(0)
        else:
            iterable = list(iterable)
            total = len(iterable)

        for text in tqdm(iterable, desc="Encode text...", total=total):
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a string."""
        tokens = [
            self.vocab.get(
                id_, "\ufffd".encode()
            )  # Use the replacement character for unknown tokens
            for id_ in ids
        ]
        """Join the tokens into a single string."""
        return (b"".join(tokens)).decode("utf-8", errors="replace")

    def _pre_tokenize(
        self,
        text: str,
    ) -> list[list[bytes]]:
        """Pre-tokenize a string into bytes, splitting on whitespace and special tokens."""

        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        if pattern:
            parts = re.split(f"({pattern})", text)
            splits = [part for part in parts if part]
        else:
            splits = [text]

        output = []
        for split in splits:
            if split in self.special_tokens:
                output.append([split.encode("utf-8")])
            else:
                for match in BPETokenizer.PAT.finditer(split):
                    output.append([b.to_bytes() for b in match.group(0).encode("utf-8")])
        return output
