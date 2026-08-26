"""Model configuration for GPT-2 style decoder-only transformers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of a GPT-2 architecture model.

    Field names follow the Hugging Face ``config.json`` for GPT-2 so a real
    checkpoint's config can be loaded directly with :meth:`from_hf_json`.
    """

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5
    eos_token_id: int | None = 50256

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def kv_bytes_per_token(self, itemsize: int = 4) -> int:
        """Bytes of K and V cache needed per token across all layers."""
        return 2 * self.n_layer * self.n_embd * itemsize

    @classmethod
    def from_hf_json(cls, path: str | Path) -> ModelConfig:
        raw = json.loads(Path(path).read_text())
        return cls(
            vocab_size=raw["vocab_size"],
            n_positions=raw["n_positions"],
            n_embd=raw["n_embd"],
            n_layer=raw["n_layer"],
            n_head=raw["n_head"],
            layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
            eos_token_id=raw.get("eos_token_id", 50256),
        )

    @classmethod
    def tiny(cls, vocab_size: int = 64, n_layer: int = 2) -> ModelConfig:
        """A very small config used by the unit tests (random weights, no download)."""
        return cls(
            vocab_size=vocab_size,
            n_positions=256,
            n_embd=32,
            n_layer=n_layer,
            n_head=4,
            eos_token_id=None,
        )
