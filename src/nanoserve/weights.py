"""Download GPT-2 checkpoints from the Hugging Face Hub and load them into NumPy.

No ``transformers``/``torch``/``safetensors`` dependency: files are fetched over
plain HTTPS and the safetensors container (an 8-byte little-endian header length,
a JSON header, then raw tensor bytes) is parsed directly with ``numpy.memmap``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nanoserve.config import ModelConfig
from nanoserve.model import GPT2, GPT2Weights, LayerWeights
from nanoserve.tokenizer import Tokenizer

HF_REPOS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "distilgpt2": "distilbert/distilgpt2",
}
FILES = ("config.json", "vocab.json", "merges.txt", "model.safetensors")

_DTYPES: dict[str, type[np.generic]] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def cache_dir() -> Path:
    """Where checkpoints live. Override with ``NANOSERVE_CACHE`` (default ``./weights``)."""
    return Path(os.environ.get("NANOSERVE_CACHE", "weights"))


def model_dir(name: str) -> Path:
    return cache_dir() / name


def download(name: str = "gpt2", quiet: bool = False) -> Path:
    """Fetch config, tokenizer and safetensors weights for ``name``; returns the dir."""
    if name not in HF_REPOS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(HF_REPOS)}")
    out = model_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = out / fname
        if dest.exists():
            continue
        url = f"https://huggingface.co/{HF_REPOS[name]}/resolve/main/{fname}"
        if not quiet:
            print(f"downloading {url}", file=sys.stderr)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.rename(dest)
    return out


def read_safetensors(path: str | Path) -> dict[str, NDArray[np.generic]]:
    """Parse a ``.safetensors`` file into memory-mapped NumPy arrays."""
    path = Path(path)
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors: dict[str, NDArray[np.generic]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = np.dtype(_DTYPES[info["dtype"]]).newbyteorder("<")
        tensors[name] = data[start:end].view(dtype).reshape(info["shape"])
    return tensors


def weights_from_state_dict(
    sd: dict[str, NDArray[np.generic]],
    config: ModelConfig,
    dtype: type[np.floating] = np.float32,
) -> GPT2Weights:
    """Map Hugging Face GPT-2 tensor names onto :class:`GPT2Weights`."""
    # Some checkpoints prefix every key with "transformer.".
    sd = {k.removeprefix("transformer."): v for k, v in sd.items()}

    def get(key: str) -> NDArray[np.floating]:
        # Copy out of the memmap: tensors in a safetensors file are not guaranteed to
        # be aligned, and unaligned arrays silently fall off the BLAS fast path.
        return np.array(sd[key], dtype=dtype, order="C", copy=True)

    layers = []
    for i in range(config.n_layer):
        p = f"h.{i}."
        layers.append(
            LayerWeights(
                ln_1_g=get(p + "ln_1.weight"), ln_1_b=get(p + "ln_1.bias"),
                attn_w=get(p + "attn.c_attn.weight"), attn_b=get(p + "attn.c_attn.bias"),
                attn_proj_w=get(p + "attn.c_proj.weight"),
                attn_proj_b=get(p + "attn.c_proj.bias"),
                ln_2_g=get(p + "ln_2.weight"), ln_2_b=get(p + "ln_2.bias"),
                fc_w=get(p + "mlp.c_fc.weight"), fc_b=get(p + "mlp.c_fc.bias"),
                fc_proj_w=get(p + "mlp.c_proj.weight"), fc_proj_b=get(p + "mlp.c_proj.bias"),
            )
        )  # fmt: skip
    return GPT2Weights(
        wte=get("wte.weight"), wpe=get("wpe.weight"), layers=layers,
        ln_f_g=get("ln_f.weight"), ln_f_b=get("ln_f.bias"),
    )  # fmt: skip


def load_model(name: str = "gpt2", dtype: type[np.floating] = np.float32) -> GPT2:
    """Load a downloaded checkpoint (call :func:`download` first)."""
    d = model_dir(name)
    if not (d / "model.safetensors").exists():
        raise FileNotFoundError(
            f"{d / 'model.safetensors'} not found; run `nanoserve download {name}` first"
        )
    config = ModelConfig.from_hf_json(d / "config.json")
    sd = read_safetensors(d / "model.safetensors")
    return GPT2(config, weights_from_state_dict(sd, config, dtype))


def load_tokenizer(name: str = "gpt2") -> Tokenizer:
    d = model_dir(name)
    return Tokenizer.from_files(d / "vocab.json", d / "merges.txt")


def is_downloaded(name: str = "gpt2") -> bool:
    return all((model_dir(name) / f).exists() for f in FILES)
