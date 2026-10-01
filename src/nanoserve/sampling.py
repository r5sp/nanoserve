"""Sampling parameters and logits -> token selection (temperature, top-k, top-p)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Probs = NDArray[np.float64]


@dataclass(frozen=True)
class SamplingParams:
    """Per-request generation settings.

    ``temperature == 0`` means greedy decoding. ``top_k == 0`` and ``top_p == 1.0``
    disable the respective filters. ``seed`` makes a request reproducible
    independently of what else is in the batch.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 16
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    n: int = 1

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.n < 1:
            raise ValueError("n must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def probs_from_logits(logits: NDArray[np.floating], params: SamplingParams) -> Probs:
    """The distribution actually sampled from, after temperature/top-k/top-p.

    Works on ``(V,)`` or ``(N, V)`` logits. For greedy decoding this is a one-hot
    distribution on the argmax, which lets speculative decoding treat greedy and
    stochastic sampling with a single acceptance rule.
    """
    x = np.asarray(logits, dtype=np.float64)
    squeeze = x.ndim == 1
    x = np.atleast_2d(x)
    V = x.shape[-1]

    if params.greedy:
        out = np.zeros_like(x)
        out[np.arange(len(x)), x.argmax(axis=-1)] = 1.0
        return out[0] if squeeze else out

    x = x / params.temperature
    if 0 < params.top_k < V:
        kth = np.partition(x, V - params.top_k, axis=-1)[:, V - params.top_k][:, None]
        x = np.where(x < kth, -np.inf, x)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)

    if params.top_p < 1.0:
        order = np.argsort(-p, axis=-1, kind="stable")
        sorted_p = np.take_along_axis(p, order, axis=-1)
        cum = np.cumsum(sorted_p, axis=-1)
        # Keep the smallest prefix whose mass reaches top_p (always >= 1 token).
        keep_sorted = (cum - sorted_p) < params.top_p
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, order, keep_sorted, axis=-1)
        p = np.where(keep, p, 0.0)
        p /= p.sum(axis=-1, keepdims=True)
    return p[0] if squeeze else p


def sample(probs: Probs, rng: np.random.Generator) -> int:
    """Draw one token id from a 1-D probability vector (inverse-CDF sampling)."""
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    idx = min(idx, len(probs) - 1)
    # Never return a zero-probability token because of floating point round-off.
    while probs[idx] == 0.0 and idx > 0:
        idx -= 1
    return idx
