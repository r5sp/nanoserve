"""Record reference GPT-2 logits from Hugging Face ``transformers`` into a small fixture.

The slow test ``tests/test_gpt2_reference.py`` compares nanoserve against these
numbers when ``transformers``/``torch`` are not installed. Requires the ``ref``
extra and downloaded weights:

    pip install -e '.[ref]' && nanoserve download gpt2
    python scripts/make_reference_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from nanoserve.weights import load_tokenizer, model_dir

PROMPT = "The paged attention algorithm stores the key-value cache in fixed-size blocks"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gpt2_reference.npz"


def main() -> None:
    tok = load_tokenizer("gpt2")
    ids = tok.encode(PROMPT)
    model = GPT2LMHeadModel.from_pretrained(str(model_dir("gpt2"))).eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float().numpy()
    np.savez_compressed(
        OUT,
        token_ids=np.asarray(ids, dtype=np.int64),
        # Store a slice of the vocabulary plus full-vocab summaries to keep the file small.
        logits_head=logits[:, :2048].astype(np.float32),
        argmax=logits.argmax(-1),
        logsumexp=np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1),
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) for {len(ids)} tokens")


if __name__ == "__main__":
    main()
