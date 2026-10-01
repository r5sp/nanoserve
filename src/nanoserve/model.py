"""GPT-2 forward pass in NumPy.

Two entry points share the same weights and layer math:

* :meth:`GPT2.forward_dense` -- the textbook reference. One sequence, full causal
  self-attention recomputed from scratch, no cache. Used as ground truth in tests.
* :meth:`GPT2.forward` -- the serving path. A *ragged batch* of sequences, each
  contributing any number of new tokens (a prefill chunk, a single decode token,
  or ``1 + k`` speculative tokens to verify). New K/V are scattered into a
  :class:`~nanoserve.kv_cache.PagedKVCache` and attention gathers each sequence's
  context through its block table.

All dense layers (QKV, output projection, MLP, LM head) run as one matmul over
the concatenated tokens of the whole batch -- this is where batching buys
throughput: the weights are streamed from memory once per step instead of once
per sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nanoserve.config import ModelConfig
from nanoserve.kv_cache import PagedKVCache

Array = NDArray[np.floating]


@dataclass
class LayerWeights:
    ln_1_g: Array
    ln_1_b: Array
    attn_w: Array  # (C, 3C), Conv1D layout: y = x @ W + b
    attn_b: Array
    attn_proj_w: Array  # (C, C)
    attn_proj_b: Array
    ln_2_g: Array
    ln_2_b: Array
    fc_w: Array  # (C, 4C)
    fc_b: Array
    fc_proj_w: Array  # (4C, C)
    fc_proj_b: Array


@dataclass
class GPT2Weights:
    wte: Array  # (V, C), tied with the LM head
    wpe: Array  # (P, C)
    layers: list[LayerWeights]
    ln_f_g: Array
    ln_f_b: Array


@dataclass
class SeqInput:
    """One sequence's contribution to a batched forward pass.

    ``token_ids`` occupy positions ``start_pos .. start_pos + len(token_ids) - 1``.
    ``slots`` must map every position ``0 .. start_pos + len(token_ids) - 1`` to a
    cache slot (positions before ``start_pos`` are already in the cache).
    Logits are returned for the last ``num_logits`` fed positions.
    """

    token_ids: Sequence[int]
    start_pos: int
    slots: NDArray[np.int64]
    num_logits: int = 1

    @property
    def ctx_len(self) -> int:
        return self.start_pos + len(self.token_ids)


def layer_norm(x: Array, g: Array, b: Array, eps: float) -> Array:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b


def gelu(x: Array) -> Array:
    # GPT-2 uses the tanh approximation ("gelu_new" in Hugging Face).
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def softmax(x: Array, axis: int = -1) -> Array:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class _BatchMeta:
    """Index arrays computed once per forward and reused by every layer."""

    write_slots: NDArray[np.int64]
    # Sequences feeding exactly one token are attended in one vectorized call.
    decode_rows: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    decode_slots: NDArray[np.int64] = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    decode_mask: NDArray[np.bool_] = field(default_factory=lambda: np.zeros((0, 0), bool))
    # Multi-token sequences (prefill chunks, speculative verification): (row offset, input).
    multi: list[tuple[int, SeqInput]] = field(default_factory=list)


class GPT2:
    def __init__(self, config: ModelConfig, weights: GPT2Weights) -> None:
        self.config = config
        self.w = weights
        self.dtype = weights.wte.dtype

    # -- construction ---------------------------------------------------------------
    @classmethod
    def random(
        cls, config: ModelConfig, seed: int = 0, dtype: type[np.floating] = np.float32
    ) -> GPT2:
        """Randomly initialised model (GPT-2 init scheme) for tests and benchmarks."""
        rng = np.random.default_rng(seed)
        C, V, P = config.n_embd, config.vocab_size, config.n_positions

        def normal(*shape: int, std: float = 0.02) -> Array:
            return (rng.standard_normal(shape) * std).astype(dtype)

        def ones(n: int) -> Array:
            return np.ones(n, dtype=dtype)

        def zeros(n: int) -> Array:
            return np.zeros(n, dtype=dtype)

        proj_std = 0.02 / np.sqrt(2 * config.n_layer)
        layers = [
            LayerWeights(
                ln_1_g=ones(C), ln_1_b=zeros(C),
                attn_w=normal(C, 3 * C), attn_b=zeros(3 * C),
                attn_proj_w=normal(C, C, std=proj_std), attn_proj_b=zeros(C),
                ln_2_g=ones(C), ln_2_b=zeros(C),
                fc_w=normal(C, 4 * C), fc_b=zeros(4 * C),
                fc_proj_w=normal(4 * C, C, std=proj_std), fc_proj_b=zeros(C),
            )
            for _ in range(config.n_layer)
        ]  # fmt: skip
        # A larger embedding scale gives the tiny test models peaky, non-uniform
        # next-token distributions, which makes sampling tests more meaningful.
        weights = GPT2Weights(
            wte=normal(V, C, std=0.5), wpe=normal(P, C, std=0.1), layers=layers,
            ln_f_g=ones(C), ln_f_b=zeros(C),
        )  # fmt: skip
        return cls(config, weights)

    def truncated(self, n_layer: int) -> GPT2:
        """A shallower model that shares this model's first ``n_layer`` blocks.

        Useful as an "early exit" draft model for speculative decoding: no extra
        weights, same tokenizer, cost proportional to ``n_layer``.
        """
        if not 1 <= n_layer <= self.config.n_layer:
            raise ValueError(f"n_layer must be in [1, {self.config.n_layer}]")
        cfg = ModelConfig(**{**self.config.__dict__, "n_layer": n_layer})
        w = GPT2Weights(
            wte=self.w.wte, wpe=self.w.wpe, layers=self.w.layers[:n_layer],
            ln_f_g=self.w.ln_f_g, ln_f_b=self.w.ln_f_b,
        )  # fmt: skip
        return GPT2(cfg, w)

    @property
    def num_params(self) -> int:
        arrays = [self.w.wte, self.w.wpe, self.w.ln_f_g, self.w.ln_f_b]
        for lw in self.w.layers:
            arrays.extend(vars(lw).values())
        return int(sum(a.size for a in arrays))

    # -- shared pieces ------------------------------------------------------------------
    def _embed(self, token_ids: NDArray[np.int64], positions: NDArray[np.int64]) -> Array:
        if positions.size and int(np.max(positions)) >= self.config.n_positions:
            raise ValueError(f"position exceeds n_positions={self.config.n_positions}")
        return self.w.wte[token_ids] + self.w.wpe[positions]

    def _qkv(self, x: Array, lw: LayerWeights) -> tuple[Array, Array, Array]:
        T = x.shape[0]
        H, D = self.config.n_head, self.config.head_dim
        h = layer_norm(x, lw.ln_1_g, lw.ln_1_b, self.config.layer_norm_epsilon)
        qkv = h @ lw.attn_w + lw.attn_b
        q, k, v = np.split(qkv, 3, axis=-1)
        return q.reshape(T, H, D), k.reshape(T, H, D), v.reshape(T, H, D)

    def _finish_layer(self, x: Array, attn: Array, lw: LayerWeights) -> Array:
        x = x + attn.reshape(x.shape) @ lw.attn_proj_w + lw.attn_proj_b
        h = layer_norm(x, lw.ln_2_g, lw.ln_2_b, self.config.layer_norm_epsilon)
        return x + gelu(h @ lw.fc_w + lw.fc_b) @ lw.fc_proj_w + lw.fc_proj_b

    def _logits(self, x: Array) -> Array:
        h = layer_norm(x, self.w.ln_f_g, self.w.ln_f_b, self.config.layer_norm_epsilon)
        return h @ self.w.wte.T

    # -- reference path -----------------------------------------------------------------
    def forward_dense(self, token_ids: Sequence[int]) -> Array:
        """Logits ``(T, V)`` for every position of one sequence. No cache."""
        ids = np.asarray(token_ids, dtype=np.int64)
        T = len(ids)
        x = self._embed(ids, np.arange(T))
        scale = 1.0 / np.sqrt(self.config.head_dim)
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        for lw in self.w.layers:
            q, k, v = self._qkv(x, lw)
            # (H, T, D) @ (H, D, T) -> (H, T, T)
            scores = np.einsum("thd,shd->hts", q, k) * scale
            scores = np.where(causal, -np.inf, scores)
            attn = np.einsum("hts,shd->thd", softmax(scores), v)
            x = self._finish_layer(x, attn, lw)
        return self._logits(x)

    # -- serving path -------------------------------------------------------------------
    def forward(self, inputs: Sequence[SeqInput], cache: PagedKVCache) -> list[Array]:
        """Run one batched step; returns ``(num_logits, V)`` logits per input."""
        if not inputs:
            return []
        ids = np.concatenate([np.asarray(s.token_ids, dtype=np.int64) for s in inputs])
        positions = np.concatenate(
            [np.arange(s.start_pos, s.ctx_len, dtype=np.int64) for s in inputs]
        )
        meta = self._build_meta(inputs)
        x = self._embed(ids, positions)
        for layer, lw in enumerate(self.w.layers):
            q, k, v = self._qkv(x, lw)
            cache.k[layer, meta.write_slots] = k
            cache.v[layer, meta.write_slots] = v
            attn = self._paged_attention(q, cache.k[layer], cache.v[layer], meta)
            x = self._finish_layer(x, attn, lw)

        # Only compute the (expensive, V-wide) LM head for rows we need.
        rows: list[int] = []
        offset = 0
        for s in inputs:
            n = len(s.token_ids)
            rows.extend(range(offset + n - s.num_logits, offset + n))
            offset += n
        logits = self._logits(x[rows]) if rows else np.zeros((0, self.config.vocab_size))
        out: list[Array] = []
        i = 0
        for s in inputs:
            out.append(logits[i : i + s.num_logits])
            i += s.num_logits
        return out

    def _build_meta(self, inputs: Sequence[SeqInput]) -> _BatchMeta:
        write: list[NDArray[np.int64]] = []
        decode_rows: list[int] = []
        decode_ctx: list[NDArray[np.int64]] = []
        multi: list[tuple[int, SeqInput]] = []
        row = 0
        for s in inputs:
            n = len(s.token_ids)
            if n == 0 or not 0 <= s.num_logits <= n:
                raise ValueError("need >= 1 token and 0 <= num_logits <= len(token_ids)")
            if len(s.slots) < s.ctx_len:
                raise ValueError("slots must cover every context position")
            write.append(s.slots[s.start_pos : s.ctx_len])
            if n == 1:
                decode_rows.append(row)
                decode_ctx.append(s.slots[: s.ctx_len])
            else:
                multi.append((row, s))
            row += n
        meta = _BatchMeta(write_slots=np.concatenate(write), multi=multi)
        if decode_rows:
            L = max(len(c) for c in decode_ctx)
            slot_mat = np.zeros((len(decode_ctx), L), dtype=np.int64)
            mask = np.ones((len(decode_ctx), L), dtype=bool)  # True = masked out
            for b, c in enumerate(decode_ctx):
                slot_mat[b, : len(c)] = c
                mask[b, : len(c)] = False
            meta.decode_rows = np.asarray(decode_rows, dtype=np.int64)
            meta.decode_slots = slot_mat
            meta.decode_mask = mask
        return meta

    def _paged_attention(self, q: Array, k_cache: Array, v_cache: Array, meta: _BatchMeta) -> Array:
        """Causal attention for a ragged batch, reading K/V through slot indices."""
        out = np.empty_like(q)
        scale = 1.0 / np.sqrt(self.config.head_dim)

        if len(meta.decode_rows):
            # Batched decode: pad every context to the longest one and mask.
            qd = q[meta.decode_rows]  # (B, H, D)
            kd = k_cache[meta.decode_slots]  # (B, L, H, D)
            vd = v_cache[meta.decode_slots]
            scores = np.einsum("bhd,blhd->bhl", qd, kd) * scale
            scores = np.where(meta.decode_mask[:, None, :], -np.inf, scores)
            out[meta.decode_rows] = np.einsum("bhl,blhd->bhd", softmax(scores), vd)

        for row, s in meta.multi:
            n = len(s.token_ids)
            kc = k_cache[s.slots[: s.ctx_len]]  # (L, H, D)
            vc = v_cache[s.slots[: s.ctx_len]]
            qs = q[row : row + n]  # (n, H, D)
            scores = np.einsum("nhd,lhd->hnl", qs, kc) * scale
            # Query i sits at absolute position start_pos + i and sees keys <= that.
            q_pos = s.start_pos + np.arange(n)[:, None]
            k_pos = np.arange(s.ctx_len)[None, :]
            scores = np.where(k_pos > q_pos, -np.inf, scores)
            out[row : row + n] = np.einsum("hnl,lhd->nhd", softmax(scores), vc)
        return out
