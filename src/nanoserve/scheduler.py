"""Iteration-level scheduler with continuous batching, chunked prefill and preemption.

Every engine step the scheduler builds a fresh batch (Orca-style iteration-level
scheduling) instead of running a fixed batch to completion:

1. **Running sequences first**, oldest first. A caught-up sequence costs one
   decode token (plus ``num_lookahead_slots`` when speculative decoding verifies
   draft tokens); a sequence still prefilling gets the next chunk of its prompt.
   If its next K/V slots cannot be allocated, the *youngest* running sequence is
   preempted (possibly the requester itself) until the allocation fits.
2. **Waiting sequences next**, oldest first, while the token budget, the
   ``max_num_seqs`` limit and free KV blocks allow. Prompts longer than the
   remaining budget are split into chunks (chunked prefill), so a long prompt
   never stalls ongoing decodes. Prefix-cache hits skip already-computed blocks.
   New sequences are not admitted in a step that had to preempt.

Preemption uses recomputation: the victim's blocks are freed and it returns to
the waiting queue keeping its generated tokens; when re-admitted it re-prefills
``prompt + output`` (cheaply, if its blocks are still in the prefix cache).
Ordering by arrival everywhere makes the policy FCFS and starvation-free: an old
request can be preempted only by an even older one.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from nanoserve.block_manager import BlockManager
from nanoserve.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = True
    num_lookahead_slots: int = 0  # set to k by speculative decoding


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int  # tokens fed to the target model this step (excluding draft tokens)
    num_lookahead: int = 0  # extra slots reserved for speculative draft tokens

    @property
    def completes(self) -> bool:
        """True if this step reaches the end of the sequence, so a token is sampled."""
        return self.seq.num_computed_tokens + self.num_tokens == self.seq.num_tokens


@dataclass
class SchedulerOutput:
    prefills: list[ScheduledSeq] = field(default_factory=list)
    decodes: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    rejected: list[Sequence] = field(default_factory=list)

    @property
    def scheduled(self) -> list[ScheduledSeq]:
        return self.prefills + self.decodes

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_tokens + s.num_lookahead for s in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager, max_model_len: int):
        self.config = config
        self.bm = block_manager
        self.max_model_len = max_model_len
        self.waiting: list[Sequence] = []  # sorted by arrival
        self.running: list[Sequence] = []  # sorted by arrival
        self.num_preemptions = 0

    # -- queue management -------------------------------------------------------------
    def add(self, seq: Sequence) -> None:
        if seq.num_tokens >= self.max_model_len:
            raise ValueError(f"prompt of {seq.num_tokens} tokens exceeds max_model_len")
        if not self.config.enable_chunked_prefill and (
            seq.num_tokens > self.config.max_num_batched_tokens
        ):
            raise ValueError("prompt exceeds max_num_batched_tokens and chunking is disabled")
        seq.status = SequenceStatus.WAITING
        bisect.insort(self.waiting, seq, key=lambda s: s.arrival)

    def add_running(self, seq: Sequence) -> None:
        """Insert an already-allocated sequence (a fork) directly into the running set."""
        seq.status = SequenceStatus.RUNNING
        bisect.insort(self.running, seq, key=lambda s: s.arrival)

    def finish(self, seq: Sequence, reason: str) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        self.bm.free(seq.seq_id)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.bm.free(seq.seq_id)
        seq.reset_for_recompute()
        bisect.insort(self.waiting, seq, key=lambda s: s.arrival)
        self.num_preemptions += 1

    def _lookahead(self, seq: Sequence) -> int:
        k = self.config.num_lookahead_slots
        if k == 0:
            return 0
        room_out = seq.params.max_tokens - seq.num_output_tokens - 1
        room_ctx = self.max_model_len - seq.num_tokens
        return max(0, min(k, room_out, room_ctx))

    # -- the scheduling decision -----------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        chunked = self.config.enable_chunked_prefill

        # Phase 1: running sequences, oldest first.
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            remaining = seq.num_uncomputed
            if remaining == 1:
                n, lookahead = 1, min(self._lookahead(seq), max(0, budget - 1))
            else:
                n, lookahead = (min(remaining, budget) if chunked else remaining), 0
            if n == 0 or n + lookahead > budget:
                i += 1  # no budget left for it this step; it stays running
                continue
            end = seq.num_computed_tokens + n + lookahead
            preempted_self = False
            while not self.bm.reserve(seq.seq_id, seq.kv_write_start, end):
                victim = self.running[-1]
                if victim is seq and len(self.running) == 1:
                    # Nothing else holds memory: this sequence can never grow.
                    self.finish(seq, "length")
                    out.rejected.append(seq)
                    preempted_self = True
                    break
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    preempted_self = True
                    break
            if preempted_self:
                break  # seq was the lowest-priority runner; nothing after it remains
            sched = ScheduledSeq(seq, n, lookahead)
            (out.decodes if remaining == 1 else out.prefills).append(sched)
            budget -= n + lookahead
            i += 1

        # Phase 2: admit waiting sequences, oldest first (FCFS, head-of-line blocking).
        if out.preempted:
            return out
        while self.waiting and len(self.running) < self.config.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            if not self.bm.block_tables.get(seq.seq_id):
                cached = self.bm.match_prefix(seq.seq_id, seq.token_ids)
                seq.num_computed_tokens = cached
            remaining = seq.num_uncomputed
            n = min(remaining, budget) if chunked else remaining
            if n > budget:
                break
            if not self.bm.reserve(seq.seq_id, seq.kv_write_start, seq.num_computed_tokens + n):
                self.bm.free(seq.seq_id)  # release any prefix-cache blocks we grabbed
                seq.num_computed_tokens = 0
                break
            self.waiting.pop(0)
            self.add_running(seq)
            sched = ScheduledSeq(seq, n)
            (out.decodes if remaining == 1 else out.prefills).append(sched)
            budget -= n
        return out
