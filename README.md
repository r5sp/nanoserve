# nanoserve

A small LLM inference engine in Python + NumPy. It does the main vLLM tricks (paged KV cache, continuous batching, speculative decoding with exact rejection sampling) and serves real GPT-2 weights on CPU.

About 2,300 lines of source including docstrings. Tests check it against reference implementations. It's a learning project, not a production server.

```
$ nanoserve generate --model gpt2 --prompt "The capital of France is" --temperature 0 --max-tokens 20
The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French
```

## running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # numpy + regex at runtime; pytest/ruff/mypy for dev

nanoserve download gpt2            # ~550 MB into ./weights (gitignored; override with NANOSERVE_CACHE)
nanoserve generate --prompt "Paged attention is" --max-tokens 40 --seed 0

# speculative decoding: gpt2-medium target, gpt2 draft
nanoserve download gpt2-medium
nanoserve generate --model gpt2-medium --draft-model gpt2 -k 4 --temperature 0 \
    --prompt "In machine learning, a transformer is"

# OpenAI-compatible server
nanoserve serve --model gpt2 --port 8000
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello, my name is", "max_tokens": 32, "stream": true, "seed": 1}'
```

## how it fits together

```
HTTP handlers --> inbox queue --> EngineWorker thread
                                      |
                                LLMEngine.step()
          scheduler -> (draft k tokens) -> one ragged batch
          -> GPT2.forward -> sample / verify -> update
                                      |
                    BlockManager <-> PagedKVCache
```

One `step()` is one iteration of continuous batching. Scheduler picks who runs and how many tokens each gets, the engine builds one ragged batch, runs one forward, samples (or verifies drafts), updates state. Requests can come and go between any two steps.

## the pieces

GPT-2 is in [`model.py`](src/nanoserve/model.py). I parse safetensors myself in [`weights.py`](src/nanoserve/weights.py) so there's no torch dependency, and the byte-level BPE tokenizer in [`tokenizer.py`](src/nanoserve/tokenizer.py) is also from scratch.

There are two forward paths. `forward_dense` is the plain one-sequence, no-cache version the tests use as ground truth. `forward` is the serving path: a ragged batch where each sequence brings a prefill chunk, one decode token, or `1 + k` tokens to verify. All the dense layers run as one matmul over the whole batch, so weights get read once per step. That's where batching gets its throughput.

Logits match HF `transformers` to about 4e-4 max abs diff in float32, same argmax everywhere. Paged path matches dense within 1e-9 in float64 on the tiny test model.

### paged KV cache

[`kv_cache.py`](src/nanoserve/kv_cache.py) and [`block_manager.py`](src/nanoserve/block_manager.py). Basically OS paging. K/V live in big `(n_layer, num_blocks * block_size, n_head, head_dim)` arrays, each sequence has a block table, and position `p` is at slot `table[p // B] * B + p % B`. Free list + refcounts. `reserve()` is atomic: either it allocates (and does any copy-on-write) or it changes nothing and returns `False`. The scheduler leans on that for admission and preemption.

Attention scatters new K/V into slots, then gathers context through the block table. Decode rows go in one padded masked call; prefill chunks and spec verification go per sequence with an offset causal mask.

Waste is at most `block_size - 1` slots per sequence.

Prefix caching: full blocks get registered under a chained `hash(parent_hash, block_tokens)`, and new prompts reuse matching blocks. Freed blocks keep their hash in an LRU free list until recycled, so finished or preempted prefixes often stay reusable. Parallel sampling (`n > 1`) prefills once and forks children that share blocks, with copy-on-write on the partly filled last block.

### scheduler

[`scheduler.py`](src/nanoserve/scheduler.py). Orca-style iteration-level scheduling with vLLM-style memory handling. Running sequences go first, oldest first. If one can't get its next slots, the youngest running sequence gets preempted (could be itself). Then waiting sequences get admitted FCFS under the token budget, `max_num_seqs`, and free blocks. Long prompts get chunked so they don't stall decodes. Nothing new gets admitted in a step that preempted.

Preemption is recompute only. The victim keeps its generated tokens and re-prefills `prompt + output` later, which is cheap if its blocks are still in the prefix cache. Everything's ordered by arrival so nothing starves. Each sequence has its own seeded RNG, so a seeded request gives the same tokens no matter what else is in the batch, preemption included (tested).

### speculative decoding

[`speculative.py`](src/nanoserve/speculative.py). Two proposers: a draft model (smaller GPT-2 with the same vocab, or the target's first N layers via `--draft-layers`), and n-gram prompt lookup which costs nothing. The draft model has its own paged cache that shares block ids with the target, so allocation, CoW and preemption just apply to both.

Target runs once on `[t_{L-1}, x1, ..., xk]`, then standard modified rejection sampling (Leviathan / Chen 2023): accept `xi` with prob `min(1, pi(xi) / qi(xi))`, on first rejection resample from `norm(max(0, pi - qi))`, bonus token if all accepted. Output is distributed exactly like the target's, with temperature/top-k/top-p applied. Greedy spec output is token-for-token identical to plain greedy.

Sampling is in [`sampling.py`](src/nanoserve/sampling.py), and both models use the same `probs_from_logits`, which is what keeps acceptance exact.

### server

[`server.py`](src/nanoserve/server.py), stdlib only. One worker thread owns the engine and loops `step()`. Handler threads just enqueue and read their own queue, so concurrent clients get batched. `POST /v1/completions`, `GET /v1/models`, `GET /health`. SSE streaming ends with `data: [DONE]`. A client disconnect aborts the request and frees its blocks.

## numbers

From [`benchmarks/bench.py`](benchmarks/bench.py) on an Apple Silicon Mac (M5, 10 cores, 16 GB), CPU only, NumPy 2.5 (Accelerate), float32, Python 3.14, real GPT-2 weights. Raw output in [`docs/results.json`](docs/results.json). These are CPU numbers, only useful for comparing configs to each other.

Batching, GPT-2 small, 32 requests (prompts 16–160 tokens, outputs 16–128, 2,257 output tokens total, greedy), all submitted at once:

| Mode | Output tok/s | vs. sequential |
|---|---:|---:|
| Sequential | 98.2 | 1.00x |
| Static batching, batches of 8 | 181.5 | 1.85x |
| Continuous, `max_num_seqs=8` | 191.9 | 1.95x |
| Continuous, `max_num_seqs=32` | 225.0 | 2.29x |

![throughput](docs/throughput.png)

Decode throughput goes from 86.6 tok/s at batch 1 to 221 at 16 sequences, then flattens (230.8 at 64). CPU is compute-bound past that.

KV memory, 64 requests replayed step by step, live tokens / reserved slots:

| Strategy | Utilization |
|---|---:|
| Paged, 16-token blocks | 95.9% |
| Contiguous, reserve `prompt + max_tokens` | 75.5% |
| Contiguous, reserve `max_model_len` (1,024) | 13.1% |

![kv memory](docs/kv_memory.png)

With 4,096 token slots (288 MB fp32 KV for GPT-2 small), paged ran up to 40 of these at once, with 27 recompute preemptions. Contiguous fits 25 or 4. The `prompt + max_tokens` baseline is generous since EOS is ignored here so `max_tokens` is exact.

Prefix caching, 16 requests sharing a 225-token prefix + 8 unique tokens, 16 output tokens, max 4 running:

| | Prefill tokens | Wall time |
|---|---:|---:|
| off | 3,728 | 7.44 s |
| on | 1,040 | 2.99 s |

First wave of 4 computes the prefix, the other 12 reuse 14 blocks and prefill 9 tokens each. 4 × 233 + 12 × 9 = 1,040.

Speculative, GPT-2 medium target, GPT-2 small or n-gram draft, batch 1, 4 prompts, 96 output tokens:

| Greedy | Accept | Tokens / target fwd | Speedup | Same output |
|---|---:|---:|---:|:---:|
| Baseline (37.6 tok/s) | – | 1.00 | 1.00x | – |
| Draft gpt2, k=2 | 84.7% | 2.63 | 1.01x | yes |
| Draft gpt2, k=4 | 74.4% | 3.76 | 1.16x | yes |
| N-gram, k=2 | 67.4% | 1.92 | 1.18x | yes |
| N-gram, k=4 | 54.1% | 2.21 | 1.10x | yes |

| Temp 0.8 | Accept | Tokens / target fwd | Speedup |
|---|---:|---:|---:|
| Baseline (37.1 tok/s) | – | 1.00 | 1.00x |
| Draft gpt2, k=2 | 63.9% | 2.23 | 0.84x |
| Draft gpt2, k=4 | 55.7% | 3.10 | 0.86x |
| N-gram, k=2 | 17.6% | 1.14 | 0.78x |
| N-gram, k=4 | 6.9% | 1.11 | 0.73x |

![speculative](docs/speculative.png)

So the algorithm is right but the wall-clock win on this machine is small, and negative when sampling. Reason: on this CPU, verifying 2, 3 or 5 tokens costs 1.86x, 1.94x and 2.13x a single-token decode (Accelerate has a fast 1-row matvec path, 2+ rows are much slower). A GPT-2 small draft step is 0.31x a medium step. So k=4 is about 4 × 0.31 + 2.13 ≈ 3.4 decode-equivalents for 3.76 tokens, roughly 1.1x, which is what I measured. On a GPU where verify ≈ one decode, the target-forward reduction would be the speedup.

## tests

```bash
pytest -q            # fast suite: 82 tests, ~6 s, random tiny models, no downloads
pytest -q -m slow    # needs `nanoserve download gpt2`; compares against Hugging Face
ruff check . && ruff format --check . && mypy
```

Main ones: paged == dense attention on shuffled blocks, a 2,000-op randomized allocator run that checks refcounts and ends with zero leaked blocks, spec greedy == plain greedy, and chi-square tests that the full engine reproduces the exact target distribution (plus negative controls that catch a verifier that accepts everything). CI runs ruff, mypy and the fast suite on 3.11, 3.12, 3.13.

## caveats

CPU and NumPy only, float32, no fused kernels. The paged gather copies K/V every layer, a real PagedAttention kernel reads in place. And this hardware isn't really the regime these techniques are for, so the relative numbers would look different on a GPU.

GPT-2 family only (1,024 context, no RoPE/GQA). Single process, one engine thread. No swap-to-host preemption. Draft-model spec and prefix caching interact conservatively (correct, saves less memory). Server is text completions only: no chat template, no logprobs, no auth.

## refs

- Kwon et al., PagedAttention, SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Yu et al., Orca, OSDI 2022. [usenix.org](https://www.usenix.org/conference/osdi22/presentation/yu)
- Leviathan, Kalman, Matias, Fast Inference from Transformers via Speculative Decoding, ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., Accelerating LLM Decoding with Speculative Sampling, 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Saxena, [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding), 2023
- Agrawal et al., SARATHI, 2023. [arXiv:2308.16369](https://arxiv.org/abs/2308.16369)
- Radford et al., GPT-2, 2019
- [vLLM](https://github.com/vllm-project/vllm), [nanoGPT](https://github.com/karpathy/nanoGPT), [picoGPT](https://github.com/jaymody/picoGPT)

MIT, see [LICENSE](LICENSE).
