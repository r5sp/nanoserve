"""Paged KV cache storage.

The cache for one model is two arrays ``k`` and ``v`` of shape
``(n_layer, num_blocks * block_size, n_head, head_dim)``. A *slot* is one token
position inside one block: ``slot = block_id * block_size + offset``. Sequences
never own contiguous memory; they own a *block table* (a list of block ids,
managed by :mod:`nanoserve.block_manager`) and a token at logical position ``p``
lives in slot ``block_table[p // block_size] * block_size + p % block_size``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nanoserve.config import ModelConfig


class PagedKVCache:
    """Physical KV storage for one model, addressed by slot index."""

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: type[np.floating] = np.float32,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (config.n_layer, num_blocks * block_size, config.n_head, config.head_dim)
        self.k: NDArray[np.floating] = np.zeros(shape, dtype=dtype)
        self.v: NDArray[np.floating] = np.zeros(shape, dtype=dtype)

    @property
    def nbytes(self) -> int:
        return int(self.k.nbytes + self.v.nbytes)

    def copy_block(self, src: int, dst: int) -> None:
        bs = self.block_size
        self.k[:, dst * bs : (dst + 1) * bs] = self.k[:, src * bs : (src + 1) * bs]
        self.v[:, dst * bs : (dst + 1) * bs] = self.v[:, src * bs : (src + 1) * bs]


def slots_for(block_table: Sequence[int], num_positions: int, block_size: int) -> NDArray[np.int64]:
    """Slot indices of logical positions ``0 .. num_positions-1`` of a sequence."""
    pos = np.arange(num_positions, dtype=np.int64)
    table = np.asarray(block_table, dtype=np.int64)
    return table[pos // block_size] * block_size + pos % block_size
