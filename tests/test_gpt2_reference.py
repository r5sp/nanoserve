"""Real GPT-2 (124M) logits must match Hugging Face transformers.

Slow/optional: needs ``nanoserve download gpt2``. Compares against a live
``transformers`` model when it is installed, and always against the recorded
fixture in ``tests/fixtures/gpt2_reference.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoserve.kv_cache import PagedKVCache, slots_for
from nanoserve.model import SeqInput
from nanoserve.weights import is_downloaded, load_model, load_tokenizer, model_dir

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_downloaded("gpt2"), reason="needs `nanoserve download gpt2`"),
]
FIXTURE = Path(__file__).parent / "fixtures" / "gpt2_reference.npz"
ATOL = 2e-3  # float32, 12 layers; observed max abs diff is ~4e-4


@pytest.fixture(scope="module")
def gpt2():
    return load_model("gpt2")


def test_matches_recorded_fixture(gpt2) -> None:
    ref = np.load(FIXTURE)
    logits = gpt2.forward_dense(ref["token_ids"].tolist())
    np.testing.assert_allclose(logits[:, :2048], ref["logits_head"], atol=ATOL)
    np.testing.assert_array_equal(logits.argmax(-1), ref["argmax"])
    lse = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
    np.testing.assert_allclose(lse, ref["logsumexp"], atol=ATOL)


def test_paged_path_matches_dense_on_real_weights(gpt2) -> None:
    ids = load_tokenizer("gpt2").encode("Paged attention gathers keys through a block table.")
    dense = gpt2.forward_dense(ids)
    cache = PagedKVCache(gpt2.config, num_blocks=8, block_size=4)
    table = [7, 2, 5, 0, 3, 6]
    out = []
    for p in range(len(ids)):  # token-by-token decode through the paged cache
        (o,) = gpt2.forward([SeqInput([ids[p]], p, slots_for(table, p + 1, 4))], cache)
        out.append(o[0])
    np.testing.assert_allclose(np.stack(out), dense, atol=ATOL)


def test_matches_live_transformers(gpt2) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hf = transformers.GPT2LMHeadModel.from_pretrained(str(model_dir("gpt2"))).eval()
    ids = load_tokenizer("gpt2").encode("Speculative decoding verifies k draft tokens at once.")
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()
    ours = gpt2.forward_dense(ids)
    assert np.abs(ours - ref).max() < ATOL
    np.testing.assert_array_equal(ours.argmax(-1), ref.argmax(-1))
