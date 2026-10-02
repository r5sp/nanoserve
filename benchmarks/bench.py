"""Benchmarks for nanoserve. Every number in the README comes from this script.

    python benchmarks/bench.py                # all suites (needs gpt2 + gpt2-medium)
    python benchmarks/bench.py --suite throughput kv prefix spec

Results are written to ``docs/results.json`` and charts to ``docs/*.png``.
All runs use real GPT-2 weights on CPU (NumPy, float32).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.kv_cache import slots_for
from nanoserve.model import GPT2, SeqInput
from nanoserve.sampling import SamplingParams
from nanoserve.speculative import SpeculativeConfig
from nanoserve.tokenizer import Tokenizer
from nanoserve.weights import download, is_downloaded, load_model, load_tokenizer

DOCS = Path(__file__).resolve().parent.parent / "docs"

CORPUS = """\
The history of computing is a story of abstraction. Early machines were programmed by \
rewiring panels; later, stored programs let the same hardware run many tasks. Operating \
systems introduced virtual memory, which lets each process believe it owns a large, \
contiguous address space while the kernel maps fixed-size pages onto scattered physical \
frames. Paging removed external fragmentation and made it cheap to share pages between \
processes with copy-on-write. Decades later the same idea reappeared in large language \
model serving, where the key-value cache of each request grows token by token and its \
final size is unknown in advance. Reserving the maximum length for every request wastes \
most of the memory, so modern engines split the cache into blocks and keep a block table \
per sequence. Combined with iteration-level scheduling, which lets new requests join a \
running batch at every step, this lets a server keep its accelerator busy with many \
concurrent requests. Speculative decoding attacks a different bottleneck: autoregressive \
generation produces one token per forward pass, but verifying several proposed tokens costs \
about the same as generating one when the model is limited by memory bandwidth.
"""

# Synthetic workload: prompt and output lengths vary per request, like real traffic.
PROMPT_LEN = (16, 160)
OUTPUT_LEN = (16, 128)


def _machine() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def make_workload(tok: Tokenizer, n: int, seed: int = 0) -> list[tuple[list[int], int]]:
    rng = np.random.default_rng(seed)
    ids = tok.encode(CORPUS)
    work = []
    for _ in range(n):
        plen = int(rng.integers(*PROMPT_LEN))
        start = int(rng.integers(0, len(ids)))
        prompt = [ids[(start + i) % len(ids)] for i in range(plen)]
        work.append((prompt, int(rng.integers(*OUTPUT_LEN))))
    return work


def _params(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)


def _engine(model: GPT2, **kw: Any) -> LLMEngine:
    cfg: dict[str, Any] = dict(block_size=16, num_blocks=1024, max_num_seqs=64,
                               max_num_batched_tokens=1024)  # fmt: skip
    cfg.update(kw)
    return LLMEngine(model, EngineConfig(**cfg))


def _run_to_completion(engine: LLMEngine, work: list[tuple[list[int], int]]) -> int:
    for prompt, m in work:
        engine.add_request(prompt, _params(m))
    produced = 0
    while engine.has_unfinished():
        produced += sum(len(o.new_token_ids) for o in engine.step())
    return produced


# -- 1. throughput ---------------------------------------------------------------------


def bench_throughput(model: GPT2, tok: Tokenizer, n_requests: int) -> dict[str, Any]:
    work = make_workload(tok, n_requests, seed=0)
    total_out = sum(m for _, m in work)
    res: dict[str, Any] = {"n_requests": n_requests, "output_tokens": total_out, "modes": {}}

    def record(name: str, seconds: float, tokens: int) -> None:
        res["modes"][name] = {"seconds": round(seconds, 2), "tok_per_s": round(tokens / seconds, 1)}
        print(f"  {name:28s} {tokens / seconds:8.1f} tok/s  ({seconds:.1f}s)")

    # Naive: one request at a time (same kernels, batch size 1).
    t = time.perf_counter()
    for prompt, m in work:
        _run_to_completion(_engine(model, max_num_seqs=1), [(prompt, m)])
    record("sequential (batch=1)", time.perf_counter() - t, total_out)

    # Static batching: fixed batches of 8 that run until their longest member finishes.
    t = time.perf_counter()
    produced = 0
    for i in range(0, len(work), 8):
        produced += _run_to_completion(_engine(model, max_num_seqs=8), work[i : i + 8])
    record("static batching (batch=8)", time.perf_counter() - t, produced)

    for max_seqs in (8, 32):
        eng = _engine(model, max_num_seqs=max_seqs)
        t = time.perf_counter()
        produced = _run_to_completion(eng, work)
        record(f"continuous (max_num_seqs={max_seqs})", time.perf_counter() - t, produced)

    # Throughput vs. concurrency (all requests arrive at once, decode-heavy).
    curve = []
    for c in (1, 2, 4, 8, 16, 32, 64):
        sub = [(p, 64) for p, _ in make_workload(tok, c, seed=1)]
        eng = _engine(model, max_num_seqs=c)
        t = time.perf_counter()
        produced = _run_to_completion(eng, sub)
        curve.append(
            {"concurrency": c, "tok_per_s": round(produced / (time.perf_counter() - t), 1)}
        )
        print(f"  concurrency {c:3d}: {curve[-1]['tok_per_s']:7.1f} tok/s")
    res["concurrency_curve"] = curve
    return res


# -- 2. KV cache memory ----------------------------------------------------------------


def bench_kv_memory(model: GPT2, tok: Tokenizer, n_requests: int) -> dict[str, Any]:
    """Replay a workload and compare, step by step, the KV slots each strategy reserves
    for the sequences currently running, against the tokens actually stored."""
    block_size, max_len = 16, model.config.n_positions
    work = make_workload(tok, n_requests, seed=2)
    eng = _engine(model, block_size=block_size, num_blocks=4096, max_num_seqs=n_requests)
    for prompt, m in work:
        eng.add_request(prompt, _params(m))
    live, paged, reserve_max_tokens, reserve_max_len = 0, 0, 0, 0
    while eng.has_unfinished():
        eng.step()
        running = eng.scheduler.running
        live += sum(s.num_computed_tokens for s in running)
        paged += eng.block_manager.num_used_blocks * block_size
        reserve_max_tokens += sum(s.num_prompt_tokens + s.params.max_tokens for s in running)
        reserve_max_len += len(running) * max_len

    # Capacity under a fixed memory budget: how many of these requests run at once?
    budget_blocks = 256  # 4096 token slots, ~302 MB of fp32 KV for GPT-2 small
    cap = _engine(model, block_size=block_size, num_blocks=budget_blocks, max_num_seqs=1000)
    peak = 0
    for prompt, m in work:
        cap.add_request(prompt, _params(m))
    while cap.has_unfinished():
        cap.step()
        peak = max(peak, len(cap.scheduler.running))
    mean_need = float(np.mean([len(p) + m for p, m in work]))
    slots = budget_blocks * block_size
    res = {
        "block_size": block_size,
        "utilization": {
            "paged": round(live / paged, 3),
            "contiguous_reserve_prompt_plus_max_tokens": round(live / reserve_max_tokens, 3),
            "contiguous_reserve_max_model_len": round(live / reserve_max_len, 3),
        },
        "fixed_budget": {
            "token_slots": slots,
            "mb_fp32": round(slots * model.config.kv_bytes_per_token() / 2**20, 1),
            "peak_concurrent_paged": peak,
            "concurrent_contiguous_max_tokens": int(slots // mean_need),
            "concurrent_contiguous_max_model_len": slots // max_len,
            "preemptions_paged": cap.stats.num_preemptions,
        },
    }
    print(f"  utilization: {res['utilization']}")
    print(f"  fixed budget: {res['fixed_budget']}")
    return res


# -- 3. prefix caching -----------------------------------------------------------------


def bench_prefix(model: GPT2, tok: Tokenizer) -> dict[str, Any]:
    system = tok.encode(CORPUS)[:256]
    rng = np.random.default_rng(3)
    work = [([*system, *[int(t) for t in rng.integers(0, 50000, 8)]], 16) for _ in range(16)]
    out: dict[str, Any] = {"shared_prefix_tokens": len(system), "n_requests": len(work)}
    for enabled in (False, True):
        eng = _engine(model, enable_prefix_caching=enabled, max_num_seqs=4)
        t = time.perf_counter()
        _run_to_completion(eng, work)
        key = "prefix_caching" if enabled else "no_prefix_caching"
        out[key] = {"seconds": round(time.perf_counter() - t, 2),
                    "prefill_tokens_computed": eng.stats.num_prefill_tokens}  # fmt: skip
        print(f"  {key}: {out[key]}")
    return out


# -- 4. speculative decoding -----------------------------------------------------------


def _forward_cost(model: GPT2, ctx: int, n_new: int, reps: int = 20) -> float:
    """Seconds for one forward that appends ``n_new`` tokens at context ``ctx``."""
    eng = _engine(model, num_blocks=128)
    cache, bs = eng.cache, 16
    table = list(range(-(-(ctx + n_new) // bs)))
    model.forward([SeqInput(list(range(ctx)), 0, slots_for(table, ctx, bs), 1)], cache)
    inp = SeqInput(list(range(n_new)), ctx, slots_for(table, ctx + n_new, bs), n_new)
    model.forward([inp], cache)
    t = time.perf_counter()
    for _ in range(reps):
        model.forward([inp], cache)
    return (time.perf_counter() - t) / reps


def bench_spec(tok: Tokenizer, n_prompts: int, max_tokens: int) -> dict[str, Any]:
    target, draft_small = load_model("gpt2-medium"), load_model("gpt2")
    prompts = [tok.encode(t) for t in [
        "The history of the Roman Empire",
        "In machine learning, a transformer is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "Paging removed external fragmentation and",
    ][:n_prompts]]  # fmt: skip

    def run(
        spec: SpeculativeConfig | None, draft: GPT2 | None, temperature: float
    ) -> dict[str, Any]:
        tokens, seconds, steps = 0, 0.0, 0
        proposed = accepted = 0
        outs = []
        for i, p in enumerate(prompts):
            eng = LLMEngine(target, EngineConfig(num_blocks=256, speculative=spec), draft)
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                ignore_eos=True, seed=i)  # fmt: skip
            t = time.perf_counter()
            outs.append(eng.generate([p], sp)[0])
            seconds += time.perf_counter() - t
            tokens += len(outs[-1])
            steps += eng.stats.num_steps
            proposed += eng.stats.num_spec_proposed
            accepted += eng.stats.num_spec_accepted
        return {"tok_per_s": round(tokens / seconds, 1),
                "tokens_per_target_forward": round(tokens / steps, 2),
                "acceptance_rate": round(accepted / proposed, 3) if proposed else None,
                "outputs": outs}  # fmt: skip

    res: dict[str, Any] = {"target": "gpt2-medium (355M)", "max_tokens": max_tokens,
                           "n_prompts": len(prompts), "runs": []}  # fmt: skip
    for temperature in (0.0, 0.8):
        base = run(None, None, temperature)
        configs: list[tuple[str, SpeculativeConfig | None, GPT2 | None]] = [
            ("baseline", None, None)
        ]
        for k in (2, 4):
            configs.append((f"draft gpt2 (124M), k={k}", SpeculativeConfig(k), draft_small))
        for k in (2, 4):
            configs.append((f"n-gram lookup, k={k}", SpeculativeConfig(k, method="ngram"), None))
        for name, spec, draft in configs:
            r = base if spec is None else run(spec, draft, temperature)
            row = {"temperature": temperature, "method": name,
                   "tok_per_s": r["tok_per_s"],
                   "speedup": round(r["tok_per_s"] / base["tok_per_s"], 2),
                   "tokens_per_target_forward": r["tokens_per_target_forward"],
                   "acceptance_rate": r["acceptance_rate"]}  # fmt: skip
            if temperature == 0.0:
                row["identical_to_baseline"] = r["outputs"] == base["outputs"]
            res["runs"].append(row)
            print(f"  T={temperature} {name:26s} {row}")

    # Why wall-clock speedup trails tokens-per-forward on this hardware.
    t1 = _forward_cost(target, 128, 1)
    res["verify_cost_vs_decode"] = {
        f"{n}_tokens": round(_forward_cost(target, 128, n) / t1, 2) for n in (2, 3, 5)
    }
    res["draft_cost_vs_target_decode"] = round(_forward_cost(draft_small, 128, 1) / t1, 2)
    print(f"  verify cost / decode cost: {res['verify_cost_vs_decode']}")
    print(f"  draft decode / target decode: {res['draft_cost_vs_target_decode']}")
    return res


# -- charts ----------------------------------------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _hbar(ax: Any, labels: list[str], values: list[float], fmt: str, color: str) -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.6)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)


def make_charts(results: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "text.color": INK})  # fmt: skip

    if "throughput" in results:
        thr = results["throughput"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
        modes = thr["modes"]
        _hbar(a, list(modes), [m["tok_per_s"] for m in modes.values()], "{:.0f}", SERIES[0])
        a.set_title(f"Throughput, {thr['n_requests']} requests of mixed length (tok/s)")
        curve = thr["concurrency_curve"]
        b.plot([c["concurrency"] for c in curve], [c["tok_per_s"] for c in curve],
               color=SERIES[0], linewidth=2, marker="o", markersize=5)  # fmt: skip
        b.set_xscale("log", base=2)
        b.set_xticks([c["concurrency"] for c in curve], [str(c["concurrency"]) for c in curve])
        b.set_xlabel("concurrent sequences", color=MUTED)
        b.set_ylim(bottom=0)
        b.set_title("Decode throughput vs. batch size (tok/s)")
        _style(b)
        fig.tight_layout()
        fig.savefig(DOCS / "throughput.png", dpi=150)
        plt.close(fig)

    if "kv_memory" in results:
        kv = results["kv_memory"]
        u = kv["utilization"]
        fig, ax = plt.subplots(figsize=(8, 2.6))
        labels = ["paged (16-token blocks)", "contiguous, reserve prompt+max_tokens",
                  "contiguous, reserve max_model_len"]  # fmt: skip
        vals = [100 * u["paged"], 100 * u["contiguous_reserve_prompt_plus_max_tokens"],
                100 * u["contiguous_reserve_max_model_len"]]  # fmt: skip
        _hbar(ax, labels, vals, "{:.1f}%", SERIES[0])
        ax.set_title("KV cache utilization (live tokens / reserved slots)")
        fig.tight_layout()
        fig.savefig(DOCS / "kv_memory.png", dpi=150)
        plt.close(fig)

    if "speculative" in results:
        runs = [r for r in results["speculative"]["runs"] if r["temperature"] == 0.0]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2))
        labels = [r["method"] for r in runs]
        _hbar(a, labels, [r["tokens_per_target_forward"] for r in runs], "{:.2f}", SERIES[0])
        a.set_title("Tokens per target forward pass (greedy)")
        _hbar(b, labels, [r["speedup"] for r in runs], "{:.2f}x", SERIES[1])
        b.set_title("Wall-clock speedup vs. baseline (greedy)")
        b.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
        fig.tight_layout()
        fig.savefig(DOCS / "speculative.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="*", default=["throughput", "kv", "prefix", "spec"])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--spec-tokens", type=int, default=96)
    ap.add_argument("--charts-only", action="store_true")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    out_path = DOCS / "results.json"
    results: dict[str, Any] = json.loads(out_path.read_text()) if out_path.exists() else {}

    if not args.charts_only:
        for name in ("gpt2", "gpt2-medium"):
            if not is_downloaded(name):
                download(name)
        tok = load_tokenizer("gpt2")
        model = load_model("gpt2")
        results["machine"] = _machine()
        if "throughput" in args.suite:
            print("throughput (gpt2, 124M)")
            results["throughput"] = bench_throughput(model, tok, args.requests)
        if "kv" in args.suite:
            print("kv memory (gpt2)")
            results["kv_memory"] = bench_kv_memory(model, tok, 64)
        if "prefix" in args.suite:
            print("prefix caching (gpt2)")
            results["prefix_caching"] = bench_prefix(model, tok)
        if "spec" in args.suite:
            print("speculative decoding (gpt2-medium target)")
            results["speculative"] = bench_spec(tok, 4, args.spec_tokens)
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {out_path}")
    make_charts(results)


if __name__ == "__main__":
    main()
