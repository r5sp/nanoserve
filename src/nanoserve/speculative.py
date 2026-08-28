"""Speculative decoding: cheap proposals, one target forward, exact rejection sampling.

Per decode step, for a sequence of length ``L`` whose K/V is valid up to ``L - 1``:

1. A proposer drafts up to ``k`` tokens ``x_1..x_k`` with distributions
   ``q_1..q_k`` (one-hot for deterministic proposers such as n-gram lookup).
2. The target model runs **once** on ``[t_{L-1}, x_1, ..., x_k]`` and returns
   distributions ``p_1..p_{k+1}`` for every position.
3. Draft tokens are accepted left to right: ``x_i`` is kept with probability
   ``min(1, p_i(x_i) / q_i(x_i))``. At the first rejection a replacement is drawn
   from the residual ``norm(max(0, p_i - q_i))`` and the rest are discarded. If
   all ``k`` are accepted, a bonus token is drawn from ``p_{k+1}``.

This is the modified rejection sampling scheme of Leviathan et al. (2023) and
Chen et al. (2023): every emitted token is distributed *exactly* according to the
target model (after temperature / top-k / top-p), whatever the draft does. A step
always emits between 1 and ``k + 1`` tokens. With greedy decoding ``p`` and ``q``
are one-hot, so the rule reduces to "accept while the draft matches the target's
argmax" and the output is token-for-token identical to plain greedy decoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from nanoserve.block_manager import BlockManager
from nanoserve.kv_cache import PagedKVCache, slots_for
from nanoserve.model import GPT2, SeqInput
from nanoserve.sampling import Probs, probs_from_logits, sample
from nanoserve.sequence import Sequence


@dataclass(frozen=True)
class SpeculativeConfig:
    num_speculative_tokens: int = 4
    method: Literal["draft_model", "ngram"] = "draft_model"
    ngram_max: int = 3
    ngram_min: int = 1

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("need 1 <= ngram_min <= ngram_max")


@dataclass
class Proposal:
    tokens: list[int]
    probs: Probs | None  # (len(tokens), V) draft distributions; None = deterministic


def rejection_sample(
    draft_tokens: list[int],
    draft_probs: Probs | None,
    target_probs: Probs,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Verify draft tokens against target distributions.

    ``target_probs`` has ``len(draft_tokens) + 1`` rows. Returns the emitted tokens
    (accepted draft tokens followed by one corrected/bonus token) and the number of
    draft tokens accepted.
    """
    k = len(draft_tokens)
    if target_probs.shape[0] != k + 1:
        raise ValueError("target_probs must have len(draft_tokens) + 1 rows")
    out: list[int] = []
    for i, x in enumerate(draft_tokens):
        p = target_probs[i]
        qx = 1.0 if draft_probs is None else float(draft_probs[i, x])
        px = float(p[x])
        if rng.random() * qx < px:  # u < p(x) / q(x)
            out.append(x)
            continue
        if draft_probs is None:
            residual = p.copy()
            residual[x] = 0.0
        else:
            residual = np.maximum(p - draft_probs[i], 0.0)
        total = residual.sum()
        # total > 0 whenever a rejection has non-zero probability; guard round-off.
        out.append(sample(residual / total if total > 0 else p, rng))
        return out, i
    out.append(sample(target_probs[k], rng))
    return out, k


class Proposer(ABC):
    @abstractmethod
    def propose(self, batch: list[tuple[Sequence, int]]) -> list[Proposal]:
        """Draft up to ``k`` tokens for each ``(sequence, k)``."""

    def on_verified(  # noqa: B027 - optional hook, intentionally a no-op by default
        self, seq: Sequence, length_before: int, proposal: Proposal, accepted: int
    ) -> None:
        """Hook to update proposer state after verification."""


class NgramProposer(Proposer):
    """Prompt-lookup decoding: if the last ``n`` tokens occurred earlier in the
    sequence, propose the tokens that followed that earlier occurrence.

    Costs no model compute at all, so any accepted token is pure speedup. It works
    well when the output copies from the context (code edits, extraction, RAG, or
    -- with GPT-2 -- its tendency to fall into repetition loops).
    """

    def __init__(self, ngram_max: int = 3, ngram_min: int = 1) -> None:
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min

    def lookup(self, tokens: list[int], k: int) -> list[int]:
        arr = np.asarray(tokens, dtype=np.int64)
        for n in range(min(self.ngram_max, len(arr) - 1), self.ngram_min - 1, -1):
            suffix = arr[-n:]
            windows = sliding_window_view(arr[:-1], n)  # excludes the suffix itself
            hits = np.flatnonzero((windows == suffix).all(axis=1))
            if hits.size:
                start = int(hits[-1]) + n  # most recent occurrence
                cont = arr[start : start + k]
                if cont.size:
                    return [int(t) for t in cont]
        return []

    def propose(self, batch: list[tuple[Sequence, int]]) -> list[Proposal]:
        return [Proposal(self.lookup(seq.token_ids, k), None) for seq, k in batch]


class DraftModelProposer(Proposer):
    """A smaller model drafts ``k`` tokens autoregressively (``k`` batched forwards).

    The draft keeps its own paged KV cache that shares block ids (and therefore
    block tables, allocation and copy-on-write) with the target's cache. Its valid
    prefix is tracked per sequence in ``Sequence.draft_num_computed``; stale draft
    K/V past that point is simply overwritten on the next step.
    """

    def __init__(self, draft: GPT2, block_manager: BlockManager) -> None:
        self.draft = draft
        self.bm = block_manager
        self.cache = PagedKVCache(
            draft.config, block_manager.allocator.num_blocks, block_manager.block_size,
            draft.dtype.type,
        )  # fmt: skip
        block_manager.attach_cache(self.cache)

    def _slots(self, seq: Sequence, n: int) -> NDArray[np.int64]:
        return slots_for(self.bm.block_tables[seq.seq_id], n, self.bm.block_size)

    def propose(self, batch: list[tuple[Sequence, int]]) -> list[Proposal]:
        active = [(i, seq, k) for i, (seq, k) in enumerate(batch) if k > 0]
        tokens: list[list[int]] = [[] for _ in batch]
        probs: list[list[Probs]] = [[] for _ in batch]
        # Pass 1: catch the draft cache up to the sequence and draft the first token.
        inputs = []
        for _, seq, _ in active:
            d, L = seq.draft_num_computed, seq.num_tokens
            inputs.append(SeqInput(seq.token_ids[d:L], d, self._slots(seq, L)))
        logits = self.draft.forward(inputs, self.cache)
        for (i, seq, _), lg in zip(active, logits, strict=True):
            seq.draft_num_computed = seq.num_tokens
            q = probs_from_logits(lg[-1], seq.params)
            tokens[i].append(sample(q, seq.rng))
            probs[i].append(q)
        # Passes 2..k: one token per still-active sequence per pass.
        for j in range(1, max((k for _, _, k in active), default=0)):
            step = [(i, seq) for i, seq, k in active if k > j]
            inputs = [
                SeqInput(
                    [tokens[i][-1]], seq.num_tokens + j - 1, self._slots(seq, seq.num_tokens + j)
                )
                for i, seq in step
            ]
            logits = self.draft.forward(inputs, self.cache)
            for (i, seq), lg in zip(step, logits, strict=True):
                seq.draft_num_computed = seq.num_tokens + j
                q = probs_from_logits(lg[-1], seq.params)
                tokens[i].append(sample(q, seq.rng))
                probs[i].append(q)
        return [
            Proposal(tokens[i], np.stack(probs[i]) if probs[i] else None) for i in range(len(batch))
        ]

    def on_verified(
        self, seq: Sequence, length_before: int, proposal: Proposal, accepted: int
    ) -> None:
        # Draft K/V for x_1..x_{k-1} was written at positions L..L+k-2; only the
        # accepted ones remain valid.
        seq.draft_num_computed = min(seq.draft_num_computed, length_before + accepted)


def make_proposer(
    config: SpeculativeConfig, block_manager: BlockManager, draft_model: GPT2 | None
) -> Proposer:
    if config.method == "ngram":
        return NgramProposer(config.ngram_max, config.ngram_min)
    if draft_model is None:
        raise ValueError("speculative method 'draft_model' requires a draft model")
    return DraftModelProposer(draft_model, block_manager)
