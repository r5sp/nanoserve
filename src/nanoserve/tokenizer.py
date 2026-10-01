"""GPT-2 byte-level BPE tokenizer, implemented from ``vocab.json`` + ``merges.txt``.

How GPT-2 BPE works:

1. Text is split into chunks by a regex (contractions, runs of letters, runs of
   digits, runs of punctuation, whitespace), each optionally with one leading space.
2. Each chunk is encoded to UTF-8 and every byte is mapped to a printable unicode
   character (``bytes_to_unicode``), so the vocabulary never contains raw control
   bytes and every string is representable -- there is no "unknown" token.
3. Within a chunk, adjacent symbol pairs are merged greedily in order of merge
   rank (lowest rank first) until no learned merge applies.
4. The resulting symbols are looked up in the vocabulary.

Decoding reverses the mapping: ids -> symbol strings -> bytes -> UTF-8.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import regex

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-character table."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return dict(zip(printable, map(chr, chars), strict=True))


class Tokenizer:
    def __init__(
        self,
        encoder: dict[str, int],
        merges: Sequence[tuple[str, str]],
        special_tokens: Iterable[str] = ("<|endoftext|>",),
    ) -> None:
        self.encoder = dict(encoder)
        self.decoder = {i: s for s, i in self.encoder.items()}
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.special_tokens = {s: self.encoder[s] for s in special_tokens if s in self.encoder}
        self._pat = regex.compile(GPT2_PATTERN)
        self._special_pat = (
            regex.compile("(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_json: str | Path, merges_txt: str | Path) -> Tokenizer:
        encoder = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
        lines = Path(merges_txt).read_text(encoding="utf-8").splitlines()
        merges = [
            (a, b) for a, b in (ln.split() for ln in lines if ln and not ln.startswith("#version"))
        ]
        return cls(encoder, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def eos_token_id(self) -> int | None:
        return self.special_tokens.get("<|endoftext|>")

    # -- encoding ---------------------------------------------------------------------
    def _bpe(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        word = [self.byte_encoder[b] for b in chunk.encode("utf-8")]
        while len(word) > 1:
            # Find the adjacent pair with the lowest merge rank.
            best_rank, best_i = None, -1
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = word[best_i], word[best_i + 1]
            # Merge every occurrence of (a, b) left to right, as the reference does.
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        ids = [self.encoder[s] for s in word]
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode text. Special tokens such as ``<|endoftext|>`` map to their id."""
        ids: list[int] = []
        parts = (
            self._special_pat.split(text)
            if allow_special and self._special_pat is not None
            else [text]
        )
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in self._pat.findall(part):
                ids.extend(self._bpe(chunk))
        return ids

    # -- decoding ---------------------------------------------------------------------
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return bytes(self.byte_decoder[c] for i in ids for c in self.decoder[i])

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


class IncrementalDetokenizer:
    """Streams text as tokens arrive without splitting multi-byte UTF-8 characters.

    A single token can end in the middle of a UTF-8 sequence (e.g. half an emoji).
    The incremental decoder buffers the incomplete tail until the next token
    completes it.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.text = ""

    def push(self, token_ids: Iterable[int]) -> str:
        delta = self._decoder.decode(self.tokenizer.decode_bytes(token_ids))
        self.text += delta
        return delta

    def flush(self) -> str:
        delta = self._decoder.decode(b"", final=True)
        self.text += delta
        return delta
