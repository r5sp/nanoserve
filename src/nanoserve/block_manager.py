"""Block allocation: a refcounting free-list allocator and per-sequence block tables.

Blocks are reference counted. Two mechanisms create shared blocks:

* :meth:`BlockManager.fork` -- a child sequence (parallel sampling, ``n > 1``)
  shares all of its parent's blocks.
* Automatic prefix caching -- every *full, computed* block is registered under a
  hash of its token contents chained with the hash of all preceding blocks. A new
  sequence whose prompt starts with the same tokens reuses those blocks instead
  of recomputing them. Freed blocks keep their hash and stay reusable until the
  allocator recycles them (LRU order).

Any write into a block whose refcount is > 1 first copies the block
(copy-on-write), so a sequence can never corrupt KV that another sequence reads.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from nanoserve.kv_cache import PagedKVCache


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Free-list allocator with reference counts and an LRU-evictable prefix cache.

    The free list is an ``OrderedDict`` used as an LRU queue: blocks are appended
    when their refcount drops to zero and popped from the front when allocated.
    A free block may still be *cached* (registered under a content hash); it is
    only evicted from the hash table when it is actually reused.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.refcount = [0] * num_blocks
        self._free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(num_blocks))
        self._hash_of: dict[int, int] = {}  # block id -> content hash
        self._block_of: dict[int, int] = {}  # content hash -> block id

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("no free KV blocks")
        block, _ = self._free.popitem(last=False)
        h = self._hash_of.pop(block, None)
        if h is not None:
            del self._block_of[h]
        self.refcount[block] = 1
        return block

    def incref(self, block: int) -> None:
        if self.refcount[block] == 0:
            # Reviving a cached-but-free block.
            del self._free[block]
        self.refcount[block] += 1

    def decref(self, block: int) -> None:
        if self.refcount[block] <= 0:
            raise RuntimeError(f"double free of block {block}")
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            self._free[block] = None

    # -- prefix cache -------------------------------------------------------------
    def lookup(self, h: int) -> int | None:
        return self._block_of.get(h)

    def register(self, block: int, h: int) -> bool:
        """Associate a computed, full block with a content hash. Returns False if taken."""
        if h in self._block_of or block in self._hash_of:
            return False
        self._block_of[h] = block
        self._hash_of[block] = h
        return True

    @property
    def num_cached(self) -> int:
        return len(self._block_of)


def block_hash(parent: int | None, tokens: Sequence[int]) -> int:
    """Content hash of a full block, chained with the hash of everything before it."""
    return hash((parent, tuple(tokens)))


class BlockManager:
    """Owns the block tables of all live sequences.

    All mutating operations are atomic: they either fully succeed or leave state
    unchanged and return ``False``. Copy-on-write copies are applied immediately
    to every attached :class:`PagedKVCache` (the target model's and, when
    speculative decoding uses a draft model, the draft's -- both share block ids).
    """

    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True
    ) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.enable_prefix_caching = enable_prefix_caching
        self.block_tables: dict[int, list[int]] = {}
        self._hashes: dict[int, list[int]] = {}  # seq id -> hashes of its registered blocks
        self.caches: list[PagedKVCache] = []
        self.num_cow_copies = 0
        self.num_prefix_hit_tokens = 0

    def attach_cache(self, cache: PagedKVCache) -> None:
        if cache.block_size != self.block_size or cache.num_blocks != self.allocator.num_blocks:
            raise ValueError("cache geometry does not match the block manager")
        self.caches.append(cache)

    # -- queries ------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    @property
    def num_used_blocks(self) -> int:
        return self.allocator.num_blocks - self.allocator.num_free

    def table(self, seq_id: int) -> list[int]:
        return self.block_tables.setdefault(seq_id, [])

    def _blocks_for(self, num_positions: int) -> int:
        return -(-num_positions // self.block_size)

    def _cow_indices(self, table: list[int], start: int, end: int) -> list[int]:
        if end <= start:
            return []
        lo = start // self.block_size
        hi = min(len(table), self._blocks_for(end))
        return [i for i in range(lo, hi) if self.allocator.refcount[table[i]] > 1]

    def blocks_needed(self, seq_id: int, write_start: int, write_end: int) -> int:
        """Free blocks :meth:`reserve` would consume for this write."""
        table = self.block_tables.get(seq_id, [])
        new = max(0, self._blocks_for(write_end) - len(table))
        return new + len(self._cow_indices(table, write_start, write_end))

    # -- mutation -----------------------------------------------------------------
    def reserve(self, seq_id: int, write_start: int, write_end: int) -> bool:
        """Make positions ``[write_start, write_end)`` writable for ``seq_id``.

        Grows the block table to cover ``write_end`` positions and copies any shared
        block in the write range (copy-on-write). Returns False without side
        effects if there are not enough free blocks.
        """
        table = self.table(seq_id)
        cow = self._cow_indices(table, write_start, write_end)
        new = max(0, self._blocks_for(write_end) - len(table))
        if new + len(cow) > self.allocator.num_free:
            return False
        for i in cow:
            old = table[i]
            fresh = self.allocator.allocate()
            for cache in self.caches:
                cache.copy_block(old, fresh)
            self.allocator.decref(old)
            table[i] = fresh
            self.num_cow_copies += 1
            hashes = self._hashes.get(seq_id)
            if hashes is not None and i < len(hashes):
                del hashes[i:]  # the copy is not registered; drop it and later hashes
        for _ in range(new):
            table.append(self.allocator.allocate())
        return True

    def fork(self, parent_id: int, child_id: int) -> None:
        """Give ``child_id`` a block table that shares every block of ``parent_id``."""
        if self.block_tables.get(child_id):
            raise ValueError(f"sequence {child_id} already has blocks")
        parent = self.block_tables[parent_id]
        for b in parent:
            self.allocator.incref(b)
        self.block_tables[child_id] = list(parent)
        self._hashes[child_id] = list(self._hashes.get(parent_id, []))

    def free(self, seq_id: int) -> None:
        # Release in reverse so a sequence's tail blocks are evicted from the
        # prefix cache before its (more shareable) head blocks.
        for b in reversed(self.block_tables.pop(seq_id, [])):
            self.allocator.decref(b)
        self._hashes.pop(seq_id, None)

    # -- prefix caching -----------------------------------------------------------
    def match_prefix(self, seq_id: int, tokens: Sequence[int]) -> int:
        """Attach cached blocks matching a prefix of ``tokens``; return #tokens reused.

        At most ``len(tokens) - 1`` tokens are reused so the last token is always
        recomputed (we need its logits).
        """
        if not self.enable_prefix_caching or self.block_tables.get(seq_id):
            return 0
        bs = self.block_size
        max_blocks = (len(tokens) - 1) // bs
        parent: int | None = None
        matched: list[int] = []
        hashes: list[int] = []
        for i in range(max_blocks):
            h = block_hash(parent, tokens[i * bs : (i + 1) * bs])
            block = self.allocator.lookup(h)
            if block is None:
                break
            matched.append(block)
            hashes.append(h)
            parent = h
        for b in matched:
            self.allocator.incref(b)
        self.block_tables[seq_id] = matched
        self._hashes[seq_id] = hashes
        n = len(matched) * bs
        self.num_prefix_hit_tokens += n
        return n

    def register_computed(self, seq_id: int, tokens: Sequence[int], num_computed: int) -> None:
        """Register every full block whose positions are all computed."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        table = self.block_tables[seq_id]
        hashes = self._hashes.setdefault(seq_id, [])
        full = min(num_computed // bs, len(table))
        while len(hashes) < full:
            i = len(hashes)
            parent = hashes[-1] if hashes else None
            h = block_hash(parent, tokens[i * bs : (i + 1) * bs])
            self.allocator.register(table[i], h)  # no-op if an identical block is cached
            hashes.append(h)

    def check_invariants(self) -> None:
        """Assert allocator/table consistency (used by tests)."""
        counts = [0] * self.allocator.num_blocks
        for table in self.block_tables.values():
            for b in table:
                counts[b] += 1
        assert counts == self.allocator.refcount, "refcounts disagree with block tables"
        free = set(self.allocator._free)
        for b, c in enumerate(counts):
            assert (c == 0) == (b in free), f"block {b}: refcount {c} but free={b in free}"
