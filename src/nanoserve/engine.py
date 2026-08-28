"""The inference engine: ties the scheduler, block manager, model and sampler together.

One call to :meth:`LLMEngine.step` is one iteration of continuous batching:

    schedule -> build ragged batch -> one model forward -> sample -> update state

Requests can be added between any two steps; finished sequences release their
KV blocks immediately so waiting requests can be admitted on the next step.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence as Seq
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nanoserve.block_manager import BlockManager
from nanoserve.kv_cache import PagedKVCache, slots_for
from nanoserve.model import GPT2, SeqInput
from nanoserve.sampling import SamplingParams, probs_from_logits, sample
from nanoserve.scheduler import ScheduledSeq, Scheduler, SchedulerConfig
from nanoserve.sequence import Sequence
from nanoserve.speculative import Proposal, SpeculativeConfig, make_proposer, rejection_sample


@dataclass
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 512
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    max_model_len: int | None = None  # defaults to the model's n_positions
    speculative: SpeculativeConfig | None = None


@dataclass
class RequestOutput:
    request_id: str
    index: int
    new_token_ids: list[int]
    output_token_ids: list[int]
    finished: bool
    finish_reason: str | None = None


@dataclass
class EngineStats:
    num_steps: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0  # tokens fed on decode steps (incl. draft tokens verified)
    num_generated_tokens: int = 0
    num_preemptions: int = 0
    num_spec_proposed: int = 0
    num_spec_accepted: int = 0
    # Per-step memory accounting: blocks in use vs. tokens actually stored.
    kv_used_blocks: list[int] = field(default_factory=list)
    kv_live_tokens: list[int] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def spec_acceptance_rate(self) -> float:
        return self.num_spec_accepted / self.num_spec_proposed if self.num_spec_proposed else 0.0


class LLMEngine:
    def __init__(
        self, model: GPT2, config: EngineConfig | None = None, draft_model: GPT2 | None = None
    ) -> None:
        self.model = model
        self.config = config or EngineConfig()
        cfg = self.config
        self.max_model_len = min(cfg.max_model_len or model.config.n_positions,
                                 model.config.n_positions)  # fmt: skip
        self.block_manager = BlockManager(cfg.num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.cache = PagedKVCache(model.config, cfg.num_blocks, cfg.block_size, model.dtype.type)
        self.block_manager.attach_cache(self.cache)
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_num_seqs=cfg.max_num_seqs,
                max_num_batched_tokens=cfg.max_num_batched_tokens,
                enable_chunked_prefill=cfg.enable_chunked_prefill,
                num_lookahead_slots=cfg.speculative.num_speculative_tokens
                if cfg.speculative
                else 0,
            ),
            self.block_manager,
            self.max_model_len,
        )
        self.proposer = None
        self.uses_draft_kv = False
        if cfg.speculative is not None:
            if draft_model is not None and draft_model.config.vocab_size != model.config.vocab_size:
                raise ValueError("draft and target models must share a vocabulary")
            self.proposer = make_proposer(cfg.speculative, self.block_manager, draft_model)
            self.uses_draft_kv = cfg.speculative.method == "draft_model"
        self.stats = EngineStats()
        self._seq_ids = itertools.count()
        self._arrivals = itertools.count()
        self._req_ids = itertools.count()
        self.sequences: dict[str, list[Sequence]] = {}

    # -- request lifecycle ------------------------------------------------------------
    def add_request(
        self,
        prompt_token_ids: Seq[int],
        params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> str:
        params = params or SamplingParams()
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if max(prompt_token_ids) >= self.model.config.vocab_size or min(prompt_token_ids) < 0:
            raise ValueError("prompt contains out-of-vocabulary token ids")
        rid = request_id if request_id is not None else f"req-{next(self._req_ids)}"
        if rid in self.sequences:
            raise ValueError(f"duplicate request id {rid!r}")
        seq = Sequence(
            seq_id=next(self._seq_ids),
            request_id=rid,
            prompt_token_ids=list(prompt_token_ids),
            params=params,
            arrival=next(self._arrivals),
            pending_forks=params.n - 1,
            uses_draft_kv=self.uses_draft_kv,
        )
        self.scheduler.add(seq)
        self.sequences[rid] = [seq]
        return rid

    def abort_request(self, request_id: str) -> None:
        for seq in self.sequences.pop(request_id, []):
            if not seq.is_finished:
                self.scheduler.finish(seq, "abort")

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # -- one iteration ----------------------------------------------------------------
    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        self.stats.num_preemptions += len(sched.preempted)
        for seq in sched.rejected:  # could not fit in the KV cache even when alone
            self._release_if_done(seq)
        outputs = [self._output(s, []) for s in sched.rejected]
        if sched.is_empty:
            return outputs

        scheduled = sched.scheduled
        proposals = self._propose(sched.decodes)
        inputs = [self._make_input(s, proposals.get(s.seq.seq_id)) for s in scheduled]
        logits = self.model.forward(inputs, self.cache)

        self.stats.num_steps += 1
        self.stats.batch_sizes.append(len(scheduled))
        self.stats.num_prefill_tokens += sum(s.num_tokens for s in sched.prefills)
        self.stats.num_decode_tokens += sum(len(i.token_ids) for i in inputs[len(sched.prefills) :])

        for s, lg in zip(scheduled, logits, strict=True):
            seq = s.seq
            completes = s.completes
            seq.num_computed_tokens += s.num_tokens
            if not completes:
                self.block_manager.register_computed(
                    seq.seq_id, seq.token_ids, seq.num_computed_tokens
                )
                continue
            prop = proposals.get(seq.seq_id)
            if prop is not None:
                outputs.append(self._verify(seq, prop, lg))
                continue
            for child in self._fork_children(seq):
                tok = sample(probs_from_logits(lg[-1], child.params), child.rng)
                outputs.append(self._append_tokens(child, [tok]))
            tok = sample(probs_from_logits(lg[-1], seq.params), seq.rng)
            outputs.append(self._append_tokens(seq, [tok]))

        self._record_kv_usage()
        return outputs

    def _make_input(self, s: ScheduledSeq, proposal: Proposal | None) -> SeqInput:
        seq = s.seq
        start = seq.num_computed_tokens
        end = start + s.num_tokens
        tokens = seq.token_ids[start:end]
        num_logits = 1 if s.completes else 0
        if proposal is not None:
            # Verify all draft tokens in the same forward: logits for every position.
            tokens = tokens + proposal.tokens
            end += len(proposal.tokens)
            num_logits = len(tokens)
        table = self.block_manager.block_tables[seq.seq_id]
        return SeqInput(tokens, start, slots_for(table, end, self.config.block_size), num_logits)

    # -- speculative decoding ---------------------------------------------------------
    def _propose(self, decodes: list[ScheduledSeq]) -> dict[int, Proposal]:
        if self.proposer is None:
            return {}
        # A parent that still has to fork its n-1 children samples its first token
        # without speculation so the children can share its prompt blocks.
        batch = [
            (s.seq, s.num_lookahead)
            for s in decodes
            if s.num_lookahead > 0 and s.seq.pending_forks == 0
        ]
        if not batch:
            return {}
        proposals = self.proposer.propose(batch)
        return {seq.seq_id: p for (seq, _), p in zip(batch, proposals, strict=True) if p.tokens}

    def _verify(self, seq: Sequence, prop: Proposal, logits: NDArray[np.floating]) -> RequestOutput:
        assert self.proposer is not None
        length_before = seq.num_tokens
        target_probs = probs_from_logits(logits, seq.params)
        tokens, accepted = rejection_sample(prop.tokens, prop.probs, target_probs, seq.rng)
        # K/V written for the accepted draft tokens is now valid; the rest is stale.
        seq.num_computed_tokens += accepted
        self.proposer.on_verified(seq, length_before, prop, accepted)
        self.stats.num_spec_proposed += len(prop.tokens)
        self.stats.num_spec_accepted += accepted
        return self._append_tokens(seq, tokens)

    def _fork_children(self, parent: Sequence) -> list[Sequence]:
        """Parallel sampling: after the prompt is prefilled once, fork n-1 children
        that share its KV blocks (copy-on-write)."""
        children = []
        for _ in range(parent.pending_forks):
            child = Sequence(
                seq_id=next(self._seq_ids),
                request_id=parent.request_id,
                prompt_token_ids=parent.prompt_token_ids,
                params=parent.params,
                arrival=parent.arrival,
                index=len(self.sequences[parent.request_id]),
                token_ids=list(parent.token_ids),
                num_computed_tokens=parent.num_computed_tokens,
                uses_draft_kv=parent.uses_draft_kv,
                draft_num_computed=parent.draft_num_computed,
                arrival_time=parent.arrival_time,
            )
            self.block_manager.fork(parent.seq_id, child.seq_id)
            self.scheduler.add_running(child)
            self.sequences[parent.request_id].append(child)
            children.append(child)
        parent.pending_forks = 0
        return children

    def _append_tokens(self, seq: Sequence, new_tokens: list[int]) -> RequestOutput:
        """Append sampled tokens, stopping at EOS / stop ids / max_tokens / max_model_len."""
        p = seq.params
        eos = self.model.config.eos_token_id
        appended: list[int] = []
        reason: str | None = None
        for tok in new_tokens:
            seq.token_ids.append(tok)
            appended.append(tok)
            if (not p.ignore_eos and tok == eos) or tok in p.stop_token_ids:
                reason = "stop"
            elif seq.num_output_tokens >= p.max_tokens or seq.num_tokens >= self.max_model_len:
                reason = "length"
            if reason:
                break
        now = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = now
        self.stats.num_generated_tokens += len(appended)
        if reason:
            self.scheduler.finish(seq, reason)
            self._release_if_done(seq)
        else:
            self.block_manager.register_computed(seq.seq_id, seq.token_ids, seq.num_computed_tokens)
        return self._output(seq, appended)

    def _release_if_done(self, seq: Sequence) -> None:
        seq.finish_time = time.perf_counter()
        group = self.sequences.get(seq.request_id, [])
        if all(s.is_finished and s.pending_forks == 0 for s in group):
            self.sequences.pop(seq.request_id, None)

    def _output(self, seq: Sequence, new_tokens: list[int]) -> RequestOutput:
        return RequestOutput(
            request_id=seq.request_id,
            index=seq.index,
            new_token_ids=new_tokens,
            output_token_ids=seq.output_token_ids,
            finished=seq.is_finished,
            finish_reason=seq.finish_reason,
        )

    def _record_kv_usage(self) -> None:
        self.stats.kv_used_blocks.append(self.block_manager.num_used_blocks)
        self.stats.kv_live_tokens.append(sum(s.num_computed_tokens for s in self.scheduler.running))

    # -- convenience ------------------------------------------------------------------
    def generate(
        self,
        prompts: Seq[Seq[int]],
        params: SamplingParams | Seq[SamplingParams] | None = None,
    ) -> list[list[int]]:
        """Run a batch of prompts to completion; returns output tokens per prompt.

        For ``n > 1`` only choice 0 is returned; use :meth:`step` for all choices.
        """
        if params is None or isinstance(params, SamplingParams):
            params = [params or SamplingParams()] * len(prompts)
        rids = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        done: dict[str, list[int]] = {}
        while self.has_unfinished():
            for out in self.step():
                if out.finished and out.index == 0:
                    done[out.request_id] = out.output_token_ids
        return [done[r] for r in rids]


def kv_utilization(stats: EngineStats, block_size: int) -> float:
    """Fraction of allocated KV slots that hold live tokens, averaged over steps."""
    used = np.asarray(stats.kv_used_blocks, dtype=np.float64) * block_size
    live = np.asarray(stats.kv_live_tokens, dtype=np.float64)
    mask = used > 0
    return float((live[mask] / used[mask]).mean()) if mask.any() else 0.0
