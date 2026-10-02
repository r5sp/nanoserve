"""Command line interface: ``nanoserve {download,generate,serve}``."""

from __future__ import annotations

import argparse
import sys
import time

from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.model import GPT2
from nanoserve.sampling import SamplingParams
from nanoserve.speculative import SpeculativeConfig
from nanoserve.tokenizer import IncrementalDetokenizer
from nanoserve.weights import HF_REPOS, download, is_downloaded, load_model, load_tokenizer


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt2", choices=sorted(HF_REPOS))
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=512, help="KV cache capacity in blocks")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--no-prefix-caching", action="store_true")
    spec = p.add_argument_group("speculative decoding")
    spec.add_argument("--draft-model", choices=sorted(HF_REPOS), help="separate draft checkpoint")
    spec.add_argument("--draft-layers", type=int, help="use the target's first N layers as draft")
    spec.add_argument("--ngram", action="store_true", help="prompt-lookup (n-gram) drafting")
    spec.add_argument("--num-speculative-tokens", "-k", type=int, default=4)


def _load(name: str) -> GPT2:
    if not is_downloaded(name):
        download(name)
    return load_model(name)


def build_engine(args: argparse.Namespace) -> LLMEngine:
    model = _load(args.model)
    draft: GPT2 | None = None
    spec: SpeculativeConfig | None = None
    k = args.num_speculative_tokens
    if args.ngram:
        spec = SpeculativeConfig(k, method="ngram")
    elif args.draft_model or args.draft_layers:
        draft = _load(args.draft_model) if args.draft_model else model.truncated(args.draft_layers)
        spec = SpeculativeConfig(k, method="draft_model")
    config = EngineConfig(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=not args.no_prefix_caching,
        speculative=spec,
    )
    return LLMEngine(model, config, draft_model=draft)


def cmd_generate(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    tok = load_tokenizer(args.model)
    params = SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=args.seed,
    )  # fmt: skip
    engine.add_request(tok.encode(args.prompt), params)
    detok = IncrementalDetokenizer(tok)
    sys.stdout.write(args.prompt)
    t0 = time.perf_counter()
    n = 0
    while engine.has_unfinished():
        for out in engine.step():
            n += len(out.new_token_ids)
            sys.stdout.write(detok.push(out.new_token_ids))
            sys.stdout.flush()
    sys.stdout.write(detok.flush() + "\n")
    dt = time.perf_counter() - t0
    msg = f"[{n} tokens in {dt:.2f}s, {n / dt:.1f} tok/s"
    if engine.proposer is not None:
        msg += f", spec acceptance {engine.stats.spec_acceptance_rate:.0%}"
    print(msg + "]", file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    from nanoserve.server import CompletionServer

    app = CompletionServer(build_engine(args), load_tokenizer(args.model), args.model)
    server = app.serve(args.host, args.port)
    print(f"nanoserve listening on http://{args.host}:{args.port}/v1/completions", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.worker.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanoserve", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch checkpoints from the Hugging Face Hub")
    d.add_argument("models", nargs="*", default=["gpt2"], help=", ".join(sorted(HF_REPOS)))

    g = sub.add_parser("generate", help="generate text for one prompt")
    _add_engine_args(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int)

    s = sub.add_parser("serve", help="run the OpenAI-compatible HTTP server")
    _add_engine_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "download":
        for name in args.models:
            print(download(name))
    elif args.command == "generate":
        cmd_generate(args)
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
