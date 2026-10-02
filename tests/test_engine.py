"""End-to-end engine behaviour: correctness under batching, chunking, preemption, sharing."""

from __future__ import annotations

import numpy as np
import pytest

from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.model import GPT2
from nanoserve.sampling import SamplingParams
from tests.helpers import dense_greedy, random_prompts

GREEDY = SamplingParams(temperature=0.0, max_tokens=12)


def _engine(model: GPT2, **kw: object) -> LLMEngine:
    cfg: dict[str, object] = dict(
        block_size=4, num_blocks=128, max_num_seqs=16, max_num_batched_tokens=64
    )
    cfg.update(kw)
    return LLMEngine(model, EngineConfig(**cfg))  # type: ignore[arg-type]


def _assert_clean(engine: LLMEngine) -> None:
    bm = engine.block_manager
    bm.check_invariants()
    assert not bm.block_tables, "finished sequences must release their block tables"
    assert bm.num_free_blocks == bm.allocator.num_blocks, "leaked KV blocks"
    assert not engine.sequences


def test_batched_greedy_matches_dense_reference(tiny_model: GPT2) -> None:
    prompts = random_prompts(6, tiny_model.config.vocab_size, 1, 30, seed=0)
    engine = _engine(tiny_model)
    outs = engine.generate(prompts, GREEDY)
    for p, o in zip(prompts, outs, strict=True):
        assert o == dense_greedy(tiny_model, p, GREEDY.max_tokens)
    assert max(engine.stats.batch_sizes) == 6  # all decoded together
    _assert_clean(engine)


def test_chunked_prefill_respects_token_budget(tiny_model: GPT2) -> None:
    prompts = [list(range(60)), [1, 2, 3]]
    engine = _engine(tiny_model, max_num_batched_tokens=16)
    budgets = []
    orig = engine.scheduler.schedule

    def spy():
        out = orig()
        budgets.append(out.num_batched_tokens)
        return out

    engine.scheduler.schedule = spy  # type: ignore[method-assign]
    outs = engine.generate(prompts, GREEDY)
    assert max(budgets) <= 16
    for p, o in zip(prompts, outs, strict=True):
        assert o == dense_greedy(tiny_model, p, GREEDY.max_tokens)
    _assert_clean(engine)


def test_decodes_continue_while_long_prompt_prefills(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model, max_num_batched_tokens=8)
    engine.add_request([5, 6, 7], SamplingParams(temperature=0, max_tokens=30))
    for _ in range(3):
        engine.step()
    engine.add_request(list(range(64)), SamplingParams(temperature=0, max_tokens=2))
    s = engine.scheduler.schedule()
    # The running decode is scheduled first; the new prompt gets the leftover budget.
    assert len(s.decodes) == 1 and len(s.prefills) == 1
    assert s.prefills[0].num_tokens == 7


def test_max_num_seqs_limits_admission(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model, max_num_seqs=3)
    for p in random_prompts(7, tiny_model.config.vocab_size, 2, 6, seed=1):
        engine.add_request(p, GREEDY)
    while engine.has_unfinished():
        engine.step()
        assert len(engine.scheduler.running) <= 3
    assert max(engine.stats.batch_sizes) == 3
    _assert_clean(engine)


def test_continuous_batching_admits_mid_flight(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model)
    a = engine.add_request([1, 2, 3], SamplingParams(temperature=0, max_tokens=20))
    for _ in range(5):
        engine.step()
    b = engine.add_request([4, 5], SamplingParams(temperature=0, max_tokens=3))
    finished: dict[str, int] = {}
    step = 0
    while engine.has_unfinished():
        step += 1
        for o in engine.step():
            if o.finished:
                finished[o.request_id] = step
    # b joined the running batch and finished long before a, without waiting for it.
    assert finished[b] < finished[a]
    _assert_clean(engine)


def test_preemption_under_memory_pressure_preserves_outputs(tiny_model: GPT2) -> None:
    prompts = random_prompts(8, tiny_model.config.vocab_size, 4, 12, seed=2)
    params = SamplingParams(temperature=0.0, max_tokens=24)
    # 12 blocks * 4 slots = 48 slots for 8 sequences that each grow to ~30 tokens.
    engine = _engine(tiny_model, num_blocks=12)
    outs = engine.generate(prompts, params)
    assert engine.stats.num_preemptions > 0
    for p, o in zip(prompts, outs, strict=True):
        assert o == dense_greedy(tiny_model, p, params.max_tokens)
    _assert_clean(engine)


def test_preemption_victims_are_youngest_and_fcfs_order_holds(tiny_model: GPT2) -> None:
    prompts = [[i + 1, i + 2, i + 3, i + 4] for i in range(6)]
    params = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True)
    engine = _engine(tiny_model, num_blocks=10)
    rids = [engine.add_request(p, params) for p in prompts]
    orig = engine.scheduler.schedule

    def spy():
        before = [s.request_id for s in engine.scheduler.running]
        out = orig()
        victims = {v.request_id for v in out.preempted}
        kept = [r for r in before if r not in victims]
        # Every victim is younger than every running sequence that was kept.
        assert all(rids.index(v) > rids.index(k) for v in victims for k in kept)
        return out

    engine.scheduler.schedule = spy  # type: ignore[method-assign]
    order = []
    while engine.has_unfinished():
        order += [o.request_id for o in engine.step() if o.finished]
    assert engine.stats.num_preemptions > 0
    # Equal-length requests finish in arrival order: no starvation, no overtaking.
    assert order == rids
    _assert_clean(engine)


def test_sequence_that_cannot_fit_is_rejected_not_livelocked(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model, num_blocks=2)  # 8 slots total
    engine.add_request([1, 2, 3], SamplingParams(temperature=0, max_tokens=50))
    outs = []
    while engine.has_unfinished():
        outs.extend(engine.step())
    assert outs[-1].finished and outs[-1].finish_reason == "length"
    # 3 prompt + 5 outputs fill all 8 slots; the 6th output is sampled but never fits.
    assert len(outs[-1].output_token_ids) == 6
    _assert_clean(engine)


def test_seeded_sampling_is_independent_of_batch_composition(tiny_model: GPT2) -> None:
    params = SamplingParams(temperature=0.9, top_k=20, top_p=0.95, max_tokens=15, seed=1234)
    alone = _engine(tiny_model).generate([[3, 1, 4, 1, 5]], params)[0]
    others = random_prompts(5, tiny_model.config.vocab_size, 2, 20, seed=3)
    engine = _engine(tiny_model, num_blocks=24)  # also forces some preemption
    mixed = engine.generate(
        [*others[:2], [3, 1, 4, 1, 5], *others[2:]],
        [SamplingParams(temperature=1.0, max_tokens=20)] * 2
        + [params]
        + [SamplingParams(temperature=1.0, max_tokens=20)] * 3,
    )[2]
    assert mixed == alone


def test_prefix_caching_reuses_blocks_and_preserves_output(tiny_model: GPT2) -> None:
    system = list(range(1, 25))  # shared 24-token "system prompt" = 6 blocks
    engine = _engine(tiny_model)
    first = engine.generate([[*system, 40, 41]], GREEDY)[0]
    second = engine.generate([[*system, 50]], GREEDY)[0]
    assert engine.block_manager.num_prefix_hit_tokens == 24
    assert first == dense_greedy(tiny_model, [*system, 40, 41], GREEDY.max_tokens)
    assert second == dense_greedy(tiny_model, [*system, 50], GREEDY.max_tokens)
    # Only the uncached suffix was fed: 26 + 11 decodes, then 1 (not 25) + 11 decodes.
    fed = engine.stats.num_prefill_tokens + engine.stats.num_decode_tokens
    assert fed == (26 + 11) + (1 + 11)
    _assert_clean(engine)


def test_parallel_sampling_forks_share_prompt_blocks(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model, enable_prefix_caching=False)
    prompt = list(range(1, 11))  # 10 tokens: the third block is partially filled
    rid = engine.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=8, n=4, seed=7))
    engine.step()  # prefill once, fork 3 children
    seqs = engine.sequences[rid]
    assert len(seqs) == 4
    tables = [engine.block_manager.block_tables[s.seq_id] for s in seqs]
    assert all(t[:2] == tables[0][:2] for t in tables)  # full prompt blocks are shared
    results: dict[int, list[int]] = {}
    while engine.has_unfinished():
        for o in engine.step():
            if o.finished:
                results[o.index] = o.output_token_ids
    assert sorted(results) == [0, 1, 2, 3]
    assert len({tuple(v) for v in results.values()}) > 1  # independent samples
    assert engine.block_manager.num_cow_copies == 3  # each writer copied the shared tail block
    _assert_clean(engine)


def test_stop_conditions(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model)
    ref = dense_greedy(tiny_model, [1, 2, 3], 10)
    stop_at = ref[4]
    out = engine.generate(
        [[1, 2, 3]], SamplingParams(temperature=0, max_tokens=10, stop_token_ids=(stop_at,))
    )[0]
    assert out == ref[: ref.index(stop_at) + 1]
    short = LLMEngine(tiny_model, EngineConfig(block_size=4, num_blocks=32, max_model_len=8))
    out = short.generate([[1, 2, 3]], SamplingParams(temperature=0, max_tokens=50))[0]
    assert len(out) == 5


def test_abort_releases_blocks(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model)
    rid = engine.add_request([1, 2, 3, 4, 5], SamplingParams(max_tokens=100))
    engine.step()
    assert engine.block_manager.num_used_blocks > 0
    engine.abort_request(rid)
    assert not engine.has_unfinished()
    _assert_clean(engine)


def test_rejects_bad_requests(tiny_model: GPT2) -> None:
    engine = _engine(tiny_model)
    with pytest.raises(ValueError):
        engine.add_request([])
    with pytest.raises(ValueError):
        engine.add_request([tiny_model.config.vocab_size])
    with pytest.raises(ValueError):
        engine.add_request(list(np.zeros(300, dtype=int)))
