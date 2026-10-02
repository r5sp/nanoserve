from __future__ import annotations

import numpy as np

from nanoserve.model import GPT2
from nanoserve.tokenizer import Tokenizer, bytes_to_unicode


def dense_greedy(model: GPT2, prompt: list[int], max_tokens: int) -> list[int]:
    """Reference generation: recompute the full sequence with the dense path each step."""
    toks = list(prompt)
    out = []
    for _ in range(max_tokens):
        nxt = int(np.argmax(model.forward_dense(toks)[-1]))
        toks.append(nxt)
        out.append(nxt)
    return out


def random_prompts(n: int, vocab: int, lo: int, hi: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(0, vocab, rng.integers(lo, hi))] for _ in range(n)]


def toy_tokenizer() -> Tokenizer:
    """Byte-level vocab (256 symbols) plus a handful of merges and <|endoftext|>."""
    encoder = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    merges = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("l", "l"), ("e", "ll"), ("Ġ", "w")]
    for a, b in merges:
        encoder[a + b] = len(encoder)
    encoder["<|endoftext|>"] = len(encoder)
    return Tokenizer(encoder, merges)
