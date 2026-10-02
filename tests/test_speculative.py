"""Speculative decoding: exact greedy equivalence and exact distribution equivalence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanoserve.config import ModelConfig
from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.model import GPT2
from nanoserve.sampling import SamplingParams, probs_from_logits
from nanoserve.speculative import NgramProposer, SpeculativeConfig, rejection_sample
from tests.helpers import dense_greedy, random_prompts


def chi2_sf(x: float, dof: int) -> float:
    """Upper tail of the chi-square distribution (Wilson-Hilferty approximation)."""
    z = ((x / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_pvalue(counts: np.ndarray, probs: np.ndarray, min_expected: float = 5.0) -> float:
    """Pearson chi-square goodness-of-fit, pooling cells with small expected counts."""
    n = counts.sum()
    expected = probs * n
    big = expected >= min_expected
    obs = [*counts[big], counts[~big].sum()]
    exp = [*expected[big], expected[~big].sum()]
    if exp[-1] < min_expected:  # fold the pooled remainder into the smallest cell
        obs, exp = obs[:-1], exp[:-1]
        obs[int(np.argmin(exp))] += counts[~big].sum()
        exp[int(np.argmin(exp))] += expected[~big].sum()
    obs_a, exp_a = np.asarray(obs, float), np.asarray(exp, float)
    stat = float(((obs_a - exp_a) ** 2 / exp_a).sum())
    return chi2_sf(stat, len(obs_a) - 1)


def _spec_engine(model: GPT2, draft: GPT2 | None, k: int = 3, method: str = "draft_model", **kw):
    cfg = dict(block_size=4, num_blocks=256, max_num_seqs=32, max_num_batched_tokens=128)
    cfg.update(kw)
    spec = SpeculativeConfig(num_speculative_tokens=k, method=method)  # type: ignore[arg-type]
    return LLMEngine(model, EngineConfig(**cfg, speculative=spec), draft_model=draft)  # type: ignore[arg-type]


# -- the acceptance rule in isolation --------------------------------------------------


def _random_dist(rng: np.random.Generator, V: int, peaked: float = 1.5) -> np.ndarray:
    p = rng.dirichlet(np.full(V, peaked))
    return p / p.sum()


@pytest.mark.parametrize("deterministic_draft", [False, True])
def test_rejection_sampling_emits_target_distribution(deterministic_draft: bool) -> None:
    rng_dist = np.random.default_rng(0)
    V, N = 6, 40_000
    p1 = _random_dist(rng_dist, V)
    q1 = _random_dist(rng_dist, V)
    p2 = _random_dist(rng_dist, V)  # second-position target, unused for first-token marginal
    rng = np.random.default_rng(123)
    counts = np.zeros(V)
    for _ in range(N):
        if deterministic_draft:
            x, q = 2, None  # n-gram style: always proposes token 2
        else:
            x = int(rng.choice(V, p=q1))
            q = q1[None, :]
        tokens, _ = rejection_sample([x], q, np.stack([p1, p2]), rng)
        counts[tokens[0]] += 1
    assert chi2_pvalue(counts, p1) > 1e-3


def test_naive_accept_all_would_fail_the_same_test() -> None:
    """Negative control: emitting draft tokens unverified yields q, not p -- the
    statistical test above has the power to detect a broken acceptance rule."""
    rng_dist = np.random.default_rng(0)
    V, N = 6, 40_000
    p1, q1 = _random_dist(rng_dist, V), _random_dist(rng_dist, V)
    rng = np.random.default_rng(123)
    counts = np.bincount(rng.choice(V, size=N, p=q1), minlength=V).astype(float)
    assert chi2_pvalue(counts, p1) < 1e-6


def test_greedy_rule_accepts_only_argmax() -> None:
    p = np.eye(5)[[3, 1, 4]]  # one-hot target distributions
    rng = np.random.default_rng(0)
    assert rejection_sample([3, 1], np.eye(5)[[3, 1]], p, rng) == ([3, 1, 4], 2)
    assert rejection_sample([3, 2], np.eye(5)[[3, 2]], p, rng) == ([3, 1], 1)
    assert rejection_sample([0, 1], None, p, rng) == ([3], 0)


# -- end to end: greedy equivalence ----------------------------------------------------


@pytest.fixture(scope="module")
def target() -> GPT2:
    return GPT2.random(ModelConfig.tiny(n_layer=3), seed=0, dtype=np.float64)


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("draft_kind", ["truncated", "unrelated"])
def test_spec_greedy_equals_plain_greedy(target: GPT2, k: int, draft_kind: str) -> None:
    draft = (
        target.truncated(1)
        if draft_kind == "truncated"
        else GPT2.random(target.config, seed=99, dtype=np.float64)
    )
    prompts = random_prompts(6, target.config.vocab_size, 1, 20, seed=k)
    params = SamplingParams(temperature=0.0, max_tokens=17)
    engine = _spec_engine(target, draft, k=k)
    outs = engine.generate(prompts, params)
    for p, o in zip(prompts, outs, strict=True):
        assert o == dense_greedy(target, p, params.max_tokens)
    assert engine.stats.num_spec_proposed > 0
    if draft_kind == "truncated":
        assert engine.stats.num_spec_accepted > 0
    engine.block_manager.check_invariants()
    assert engine.block_manager.num_free_blocks == engine.block_manager.allocator.num_blocks


def test_spec_greedy_equivalence_survives_preemption_and_chunking(target: GPT2) -> None:
    prompts = random_prompts(8, target.config.vocab_size, 5, 30, seed=7)
    params = SamplingParams(temperature=0.0, max_tokens=20)
    engine = _spec_engine(
        target, target.truncated(2), k=4, num_blocks=20, max_num_batched_tokens=16
    )
    outs = engine.generate(prompts, params)
    assert engine.stats.num_preemptions > 0
    for p, o in zip(prompts, outs, strict=True):
        assert o == dense_greedy(target, p, params.max_tokens)
    assert engine.block_manager.num_free_blocks == engine.block_manager.allocator.num_blocks


def test_ngram_spec_greedy_equals_plain_greedy(target: GPT2) -> None:
    # Repetitive prompts give the n-gram proposer something to find.
    prompts = [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2], [7, 7, 7, 7, 7], [5, 9, 5, 9, 5]]
    params = SamplingParams(temperature=0.0, max_tokens=24)
    engine = _spec_engine(target, None, k=4, method="ngram")
    outs = engine.generate(prompts, params)
    for p, o in zip(prompts, outs, strict=True):
        assert o == dense_greedy(target, p, params.max_tokens)
    assert engine.stats.num_spec_accepted > 0


def test_spec_respects_max_tokens_and_stop(target: GPT2) -> None:
    ref = dense_greedy(target, [1, 2, 3], 30)
    engine = _spec_engine(target, target.truncated(2), k=5)
    for m in [1, 2, 3, 7]:
        assert (
            engine.generate([[1, 2, 3]], SamplingParams(temperature=0, max_tokens=m))[0] == ref[:m]
        )
    stop = ref[9]
    out = engine.generate(
        [[1, 2, 3]], SamplingParams(temperature=0, max_tokens=30, stop_token_ids=(stop,))
    )[0]
    assert out == ref[: ref.index(stop) + 1]


def test_ngram_lookup() -> None:
    prop = NgramProposer(ngram_max=3)
    assert prop.lookup([1, 2, 3, 9, 1, 2, 3], k=2) == [9, 1]
    assert prop.lookup([5, 6, 7, 8], k=3) == []
    assert prop.lookup([4, 4, 4], k=2) == [4]  # no earlier trigram; bigram [4, 4] matches
    assert prop.lookup([1, 2, 1, 2], k=4) == [1, 2]


# -- end to end: distribution equivalence ----------------------------------------------


def _end_to_end_pvalue() -> tuple[float, float]:
    """Sample many 3-token continuations with speculative decoding and compare the
    empirical joint distribution against the exact target distribution, enumerated
    with the dense reference model. The draft model is unrelated to the target, so
    many proposals are rejected and the residual path is exercised heavily."""
    cfg = ModelConfig(vocab_size=6, n_positions=32, n_embd=16, n_layer=2, n_head=2,
                      eos_token_id=None)  # fmt: skip
    target = GPT2.random(cfg, seed=1, dtype=np.float64)
    draft = GPT2.random(cfg, seed=2, dtype=np.float64)
    params = SamplingParams(temperature=1.0, top_k=5, top_p=0.95, max_tokens=3)
    prompt = [0, 3, 1]
    V = cfg.vocab_size

    # Exact P(t1, t2, t3 | prompt) under the processed target distribution.
    exact = np.zeros((V, V, V))
    p1 = probs_from_logits(target.forward_dense(prompt)[-1], params)
    for a in range(V):
        if p1[a] == 0:
            continue
        p2 = probs_from_logits(target.forward_dense([*prompt, a])[-1], params)
        for b in range(V):
            if p2[b] == 0:
                continue
            p3 = probs_from_logits(target.forward_dense([*prompt, a, b])[-1], params)
            exact[a, b] = p1[a] * p2[b] * p3

    N = 12_000
    engine = _spec_engine(target, draft, k=2, num_blocks=2048, max_num_seqs=512,
                          max_num_batched_tokens=4096, enable_prefix_caching=False)  # fmt: skip
    counts = np.zeros((V, V, V))
    for i in range(N):
        engine.add_request(prompt, SamplingParams(**{**params.__dict__, "seed": i}))
    while engine.has_unfinished():
        for o in engine.step():
            if o.finished:
                a, b, c = o.output_token_ids
                counts[a, b, c] += 1
    assert engine.stats.num_spec_proposed > 0
    return chi2_pvalue(counts.ravel(), exact.ravel()), engine.stats.spec_acceptance_rate


def test_spec_sampling_matches_target_distribution_end_to_end() -> None:
    pvalue, rate = _end_to_end_pvalue()
    assert 0.05 < rate < 0.95, rate  # both the accept and the residual path are exercised
    assert pvalue > 1e-3


def test_end_to_end_test_detects_a_broken_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: a verifier that skips the acceptance test emits a mix of the
    draft's distribution; the same statistical test must reject it."""

    def accept_all(draft_tokens, draft_probs, target_probs, rng):
        from nanoserve.sampling import sample

        return [*draft_tokens, sample(target_probs[-1], rng)], len(draft_tokens)

    monkeypatch.setattr("nanoserve.engine.rejection_sample", accept_all)
    pvalue, _ = _end_to_end_pvalue()
    assert pvalue < 1e-6
