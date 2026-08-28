"""Per-sequence generation state shared by the scheduler and the engine."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import numpy as np

from nanoserve.sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    """One generated continuation. A request with ``n > 1`` owns ``n`` sequences.

    The KV invariant the engine maintains: positions ``0 .. num_computed_tokens - 1``
    of ``token_ids`` have valid K/V in the target model's paged cache. A sequence
    that is fully caught up has ``num_computed_tokens == len(token_ids) - 1``: the
    newest sampled token is fed on the next step.
    """

    seq_id: int
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival: int  # monotonically increasing admission priority (lower = older)
    index: int = 0  # choice index within the request
    token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    # Speculative decoding with a draft model: K/V valid in the *draft* cache.
    draft_num_computed: int = 0
    uses_draft_kv: bool = False
    finish_reason: str | None = None
    num_preemptions: int = 0
    pending_forks: int = 0
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not self.token_ids:
            self.token_ids = list(self.prompt_token_ids)
        seed = self.params.seed
        self.rng = np.random.default_rng(None if seed is None else [seed, self.index])

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def kv_write_start(self) -> int:
        """First position any model may write K/V for on the next step."""
        if self.uses_draft_kv:
            return min(self.num_computed_tokens, self.draft_num_computed)
        return self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def reset_for_recompute(self) -> None:
        """Preemption by recomputation: drop all KV, keep generated tokens."""
        self.num_computed_tokens = 0
        self.draft_num_computed = 0
        self.status = SequenceStatus.WAITING
        self.num_preemptions += 1
