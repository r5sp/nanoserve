from __future__ import annotations

import numpy as np
import pytest

from nanoserve.sampling import SamplingParams, probs_from_logits, sample


def test_greedy_is_one_hot_argmax() -> None:
    p = probs_from_logits(np.array([0.1, 3.0, -1.0, 2.9]), SamplingParams(temperature=0))
    np.testing.assert_array_equal(p, [0, 1, 0, 0])


def test_temperature_scales_logits() -> None:
    logits = np.array([1.0, 2.0, 3.0])
    p = probs_from_logits(logits, SamplingParams(temperature=0.5))
    e = np.exp(logits / 0.5)
    np.testing.assert_allclose(p, e / e.sum())
    hot = probs_from_logits(logits, SamplingParams(temperature=5.0))
    assert float(np.max(hot)) < float(np.max(p))  # higher temperature flattens the distribution


def test_top_k_keeps_k_largest() -> None:
    logits = np.array([0.0, 5.0, 1.0, 4.0, 3.0])
    p = probs_from_logits(logits, SamplingParams(top_k=2))
    assert set(np.flatnonzero(p)) == {1, 3}
    np.testing.assert_allclose(p.sum(), 1.0)


def test_top_p_keeps_smallest_prefix_reaching_p() -> None:
    probs = np.array([0.05, 0.5, 0.15, 0.3])
    p = probs_from_logits(np.log(probs), SamplingParams(top_p=0.7))
    # 0.5 alone < 0.7, 0.5 + 0.3 >= 0.7 -> keep exactly those two.
    np.testing.assert_allclose(p, [0, 0.5 / 0.8, 0, 0.3 / 0.8])
    p = probs_from_logits(np.log(probs), SamplingParams(top_p=1e-6))
    np.testing.assert_array_equal(p, [0, 1, 0, 0])  # always at least one token


def test_batched_rows_processed_independently() -> None:
    logits = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    p = probs_from_logits(logits, SamplingParams(top_k=1))
    np.testing.assert_array_equal(p, [[0, 0, 1], [1, 0, 0]])


def test_sample_matches_distribution_and_skips_zeros() -> None:
    probs = np.array([0.2, 0.0, 0.5, 0.3, 0.0])
    rng = np.random.default_rng(0)
    counts = np.bincount([sample(probs, rng) for _ in range(20_000)], minlength=5)
    assert counts[1] == counts[4] == 0
    np.testing.assert_allclose(counts / counts.sum(), probs, atol=0.015)


def test_seeded_rng_is_reproducible() -> None:
    probs = np.full(10, 0.1)
    a = [sample(probs, np.random.default_rng(42)) for _ in range(5)]
    b = [sample(probs, np.random.default_rng(42)) for _ in range(5)]
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -1}, {"top_k": -1}, {"top_p": 0}, {"top_p": 1.5}, {"max_tokens": 0}, {"n": 0}],
)
def test_invalid_params_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)  # type: ignore[arg-type]
