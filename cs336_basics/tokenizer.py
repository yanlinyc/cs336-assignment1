from typing import Self, Iterable, Iterator

import regex as re

from cs336_basics.io_utils import load_pickle


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    @classmethod
    def from_files(
        cls: type[Self],
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Load a BPE tokenizer from files."""
        vocab = load_pickle(vocab_filepath)
        merges = load_pickle(merges_filepath)
        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
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
        # from tqdm.auto import tqdm

        # num_lines = sum(1 for _ in iterable)
        # reset the iterator to the beginning
        # iterable.seek(0)
        # for text in tqdm(iterable, desc="Encoding texts", total=num_lines):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a string."""
        tokens = [
            self.vocab.get(
                id_, "\ufffd".encode("utf-8")
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
                for match in re.finditer(BPETokenizer.PAT, split):
                    output.append([b.to_bytes() for b in match.group(0).encode("utf-8")])
        return output
