"""Allocator / block table invariants: no leaks, correct refcounts, CoW, prefix caching."""

from __future__ import annotations

import numpy as np
import pytest

from nanoserve.block_manager import BlockAllocator, BlockManager, OutOfBlocksError
from nanoserve.config import ModelConfig
from nanoserve.kv_cache import PagedKVCache


def test_allocator_exhaustion_and_double_free() -> None:
    a = BlockAllocator(3)
    blocks = [a.allocate() for _ in range(3)]
    assert sorted(blocks) == [0, 1, 2]
    with pytest.raises(OutOfBlocksError):
        a.allocate()
    a.decref(blocks[0])
    assert a.num_free == 1
    with pytest.raises(RuntimeError):
        a.decref(blocks[0])


def test_reserve_grows_table_and_free_returns_everything() -> None:
    bm = BlockManager(num_blocks=10, block_size=4)
    assert bm.reserve(1, 0, 9)  # 9 positions -> 3 blocks
    assert len(bm.block_tables[1]) == 3
    assert bm.reserve(1, 9, 12)  # still fits in block 3
    assert len(bm.block_tables[1]) == 3
    assert bm.reserve(1, 12, 13)
    assert len(bm.block_tables[1]) == 4
    bm.check_invariants()
    bm.free(1)
    assert bm.num_free_blocks == 10
    bm.check_invariants()


def test_reserve_is_atomic_when_out_of_blocks() -> None:
    bm = BlockManager(num_blocks=4, block_size=4)
    assert bm.reserve(1, 0, 12)
    before = list(bm.block_tables[1])
    assert not bm.reserve(2, 0, 8)  # needs 2, only 1 free
    assert bm.block_tables.get(2, []) == []
    assert bm.block_tables[1] == before
    assert bm.num_free_blocks == 1
    bm.check_invariants()


def test_fork_shares_blocks_and_copy_on_write_isolates_writes() -> None:
    cfg = ModelConfig.tiny()
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    cache = PagedKVCache(cfg, 8, 4)
    bm.attach_cache(cache)
    assert bm.reserve(1, 0, 6)  # 2 blocks: one full, one half-full
    parent = list(bm.block_tables[1])
    cache.k[:] = np.random.default_rng(0).standard_normal(cache.k.shape)

    bm.fork(1, 2)
    assert bm.block_tables[2] == parent
    assert all(bm.allocator.refcount[b] == 2 for b in parent)
    bm.check_invariants()

    # Child writes position 6 (inside the shared partial block): that block is copied.
    assert bm.reserve(2, 6, 7)
    child = bm.block_tables[2]
    assert child[0] == parent[0]  # full block still shared
    assert child[1] != parent[1]  # partial block was copied
    assert bm.allocator.refcount[parent[0]] == 2
    assert bm.allocator.refcount[parent[1]] == 1
    assert bm.num_cow_copies == 1
    bs = 4
    np.testing.assert_array_equal(
        cache.k[:, child[1] * bs : (child[1] + 1) * bs],
        cache.k[:, parent[1] * bs : (parent[1] + 1) * bs],
    )
    # Parent can now write its own block in place, no further copy.
    assert bm.reserve(1, 6, 7)
    assert bm.block_tables[1] == parent
    assert bm.num_cow_copies == 1
    bm.check_invariants()
    bm.free(1)
    bm.free(2)
    assert bm.num_free_blocks == 8
    bm.check_invariants()


def test_cow_needs_free_block() -> None:
    bm = BlockManager(num_blocks=2, block_size=4, enable_prefix_caching=False)
    assert bm.reserve(1, 0, 6)
    bm.fork(1, 2)
    assert bm.blocks_needed(2, 6, 7) == 1
    assert not bm.reserve(2, 6, 7)  # no free block to copy into
    bm.check_invariants()


def test_prefix_cache_hit_reuses_full_blocks() -> None:
    bm = BlockManager(num_blocks=16, block_size=4)
    prompt = list(range(10))  # 2 full blocks + 2 tokens
    assert bm.match_prefix(1, prompt) == 0
    assert bm.reserve(1, 0, 10)
    bm.register_computed(1, prompt, 10)
    a_table = list(bm.block_tables[1])

    same_prefix = [*prompt[:8], 99, 98, 97]
    assert bm.match_prefix(2, same_prefix) == 8
    assert bm.block_tables[2] == a_table[:2]
    assert bm.allocator.refcount[a_table[0]] == 2
    bm.check_invariants()

    # Divergence in the first block means nothing is reusable.
    assert bm.match_prefix(3, [42, *prompt[1:]]) == 0


def test_prefix_cache_never_reuses_the_last_token() -> None:
    bm = BlockManager(num_blocks=16, block_size=4)
    prompt = list(range(8))  # exactly 2 full blocks
    bm.reserve(1, 0, 8)
    bm.register_computed(1, prompt, 8)
    # Reusing both blocks would leave nothing to compute logits from.
    assert bm.match_prefix(2, prompt) == 4


def test_freed_blocks_stay_cached_until_evicted() -> None:
    bm = BlockManager(num_blocks=4, block_size=2)
    prompt = [1, 2, 3, 4, 5]
    bm.reserve(1, 0, 5)
    bm.register_computed(1, prompt, 5)
    bm.free(1)
    assert bm.num_free_blocks == 4
    assert bm.allocator.num_cached == 2
    assert bm.match_prefix(2, prompt) == 4  # revived from the free list
    assert bm.num_free_blocks == 2
    bm.free(2)
    # Allocating everything recycles the cached blocks and evicts their hashes.
    assert bm.reserve(3, 0, 8)
    assert bm.allocator.num_cached == 0
    assert bm.match_prefix(4, prompt) == 0
    bm.check_invariants()


def test_randomized_operations_preserve_invariants() -> None:
    rng = np.random.default_rng(0)
    bm = BlockManager(num_blocks=32, block_size=4)
    lengths: dict[int, int] = {}
    tokens: dict[int, list[int]] = {}
    next_id = 0
    for _ in range(2000):
        op = rng.integers(0, 4)
        if op == 0 or not lengths:  # new sequence, possibly sharing a prefix
            sid, next_id = next_id, next_id + 1
            toks = [int(t) for t in rng.integers(0, 3, rng.integers(1, 20))]
            n = bm.match_prefix(sid, toks)
            if bm.reserve(sid, n, len(toks)):
                lengths[sid], tokens[sid] = len(toks), toks
                bm.register_computed(sid, toks, len(toks))
            else:
                bm.free(sid)
        elif op == 1:  # append tokens
            sid = int(rng.choice(list(lengths)))
            k = int(rng.integers(1, 6))
            if bm.reserve(sid, lengths[sid], lengths[sid] + k):
                tokens[sid] += [int(t) for t in rng.integers(0, 3, k)]
                lengths[sid] += k
                bm.register_computed(sid, tokens[sid], lengths[sid])
        elif op == 2:  # fork
            sid = int(rng.choice(list(lengths)))
            child, next_id = next_id, next_id + 1
            bm.fork(sid, child)
            lengths[child], tokens[child] = lengths[sid], list(tokens[sid])
        else:  # free
            sid = int(rng.choice(list(lengths)))
            bm.free(sid)
            del lengths[sid], tokens[sid]
        bm.check_invariants()
    for sid in list(lengths):
        bm.free(sid)
    bm.check_invariants()
    assert bm.num_free_blocks == 32
