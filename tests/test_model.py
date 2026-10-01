"""Paged, batched attention must match the dense reference forward pass."""

from __future__ import annotations

import numpy as np
import pytest

from nanoserve.config import ModelConfig
from nanoserve.kv_cache import PagedKVCache, slots_for
from nanoserve.model import GPT2, SeqInput, gelu, layer_norm


def _random_tables(n_seqs: int, max_len: int, block_size: int, num_blocks: int, rng):
    """Scattered, non-contiguous block tables (blocks drawn from a shuffled pool)."""
    pool = list(rng.permutation(num_blocks))
    per_seq = -(-max_len // block_size)
    return [[int(pool.pop()) for _ in range(per_seq)] for _ in range(n_seqs)]


def test_dense_forward_shapes(tiny_model: GPT2) -> None:
    logits = tiny_model.forward_dense([1, 2, 3, 4])
    assert logits.shape == (4, tiny_model.config.vocab_size)
    assert np.isfinite(logits).all()


def test_dense_is_causal(tiny_model: GPT2) -> None:
    a = tiny_model.forward_dense([5, 6, 7, 8, 9])
    b = tiny_model.forward_dense([5, 6, 7, 1, 1])
    np.testing.assert_allclose(a[:3], b[:3], atol=1e-12)
    assert not np.allclose(a[3], b[3])


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_paged_prefill_then_decode_matches_dense(tiny_model: GPT2, block_size: int) -> None:
    """Batched prefill of ragged prompts followed by batched decode == dense, per position."""
    rng = np.random.default_rng(1)
    cfg = tiny_model.config
    lengths = [1, 7, 13, 30]
    n_decode = 6
    seqs = [list(rng.integers(0, cfg.vocab_size, n + n_decode)) for n in lengths]
    num_blocks = 64 * (16 // block_size + 1)
    cache = PagedKVCache(cfg, num_blocks, block_size, dtype=np.float64)
    tables = _random_tables(len(seqs), max(lengths) + n_decode, block_size, num_blocks, rng)

    # Prefill all prompts in one ragged batch, asking for logits at every position.
    inputs = [
        SeqInput(s[:n], 0, slots_for(t, n, block_size), num_logits=n)
        for s, n, t in zip(seqs, lengths, tables, strict=True)
    ]
    outs = tiny_model.forward(inputs, cache)
    for s, n, out in zip(seqs, lengths, outs, strict=True):
        ref = tiny_model.forward_dense(s[:n])
        np.testing.assert_allclose(out, ref, atol=1e-9)

    # Then decode one token per sequence per step, all sequences batched together.
    for step in range(n_decode):
        inputs = [
            SeqInput([s[n + step]], n + step, slots_for(t, n + step + 1, block_size))
            for s, n, t in zip(seqs, lengths, tables, strict=True)
        ]
        outs = tiny_model.forward(inputs, cache)
        for s, n, out in zip(seqs, lengths, outs, strict=True):
            ref = tiny_model.forward_dense(s[: n + step + 1])[-1]
            np.testing.assert_allclose(out[0], ref, atol=1e-9)


def test_chunked_prefill_and_multi_token_append(tiny_model: GPT2) -> None:
    """Feeding a sequence in uneven chunks (as chunked prefill / spec verify do) == dense."""
    rng = np.random.default_rng(2)
    cfg = tiny_model.config
    bs = 4
    seq = list(rng.integers(0, cfg.vocab_size, 40))
    cache = PagedKVCache(cfg, 32, bs, dtype=np.float64)
    table = [int(b) for b in rng.permutation(32)[:10]]
    ref = tiny_model.forward_dense(seq)
    pos = 0
    for chunk in [5, 1, 9, 3, 1, 1, 20]:
        inp = SeqInput(seq[pos : pos + chunk], pos, slots_for(table, pos + chunk, bs), chunk)
        (out,) = tiny_model.forward([inp], cache)
        np.testing.assert_allclose(out, ref[pos : pos + chunk], atol=1e-9)
        pos += chunk


def test_mixed_prefill_and_decode_batch(tiny_model: GPT2) -> None:
    rng = np.random.default_rng(3)
    cfg = tiny_model.config
    bs = 8
    cache = PagedKVCache(cfg, 16, bs, dtype=np.float64)
    a = list(rng.integers(0, cfg.vocab_size, 12))
    b = list(rng.integers(0, cfg.vocab_size, 20))
    ta, tb = [0, 5], [9, 2, 7]
    tiny_model.forward([SeqInput(a[:11], 0, slots_for(ta, 11, bs))], cache)
    # Sequence a decodes its 12th token while sequence b prefills, in the same step.
    oa, ob = tiny_model.forward(
        [SeqInput(a[11:], 11, slots_for(ta, 12, bs)), SeqInput(b, 0, slots_for(tb, 20, bs))],
        cache,
    )
    np.testing.assert_allclose(oa[0], tiny_model.forward_dense(a)[-1], atol=1e-9)
    np.testing.assert_allclose(ob[0], tiny_model.forward_dense(b)[-1], atol=1e-9)


def test_truncated_model_shares_weights(tiny_model: GPT2) -> None:
    draft = tiny_model.truncated(1)
    assert draft.config.n_layer == 1
    assert draft.w.wte is tiny_model.w.wte
    assert draft.w.layers[0] is tiny_model.w.layers[0]


def test_float32_paged_close_to_float64() -> None:
    cfg = ModelConfig.tiny()
    m32 = GPT2.random(cfg, seed=0, dtype=np.float32)
    m64 = GPT2.random(cfg, seed=0, dtype=np.float64)
    toks = [3, 1, 4, 1, 5, 9, 2, 6]
    np.testing.assert_allclose(m32.forward_dense(toks), m64.forward_dense(toks), atol=1e-4)


def test_layer_norm_and_gelu() -> None:
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = layer_norm(x, np.ones(4), np.zeros(4), 1e-5)
    np.testing.assert_allclose(y.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.std(), 1.0, atol=1e-4)
    np.testing.assert_allclose(gelu(np.array([0.0])), [0.0])
    assert gelu(np.array([3.0]))[0] == pytest.approx(2.9964, abs=1e-3)


def test_position_overflow_raises(tiny_model: GPT2) -> None:
    with pytest.raises(ValueError):
        tiny_model.forward_dense(list(range(tiny_model.config.n_positions + 1)))
