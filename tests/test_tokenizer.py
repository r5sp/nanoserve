from __future__ import annotations

import pytest

from nanoserve.tokenizer import IncrementalDetokenizer, Tokenizer, bytes_to_unicode
from nanoserve.weights import is_downloaded, load_tokenizer


def _toy_tokenizer() -> Tokenizer:
    """Byte-level vocab (256 symbols) plus a handful of merges."""
    b2u = bytes_to_unicode()
    encoder = {c: i for i, c in enumerate(b2u.values())}
    merges = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("l", "l"), ("e", "ll"), ("Ġ", "w")]
    for a, b in merges:
        encoder[a + b] = len(encoder)
    encoder["<|endoftext|>"] = len(encoder)
    return Tokenizer(encoder, merges)


def test_bytes_to_unicode_is_a_bijection() -> None:
    m = bytes_to_unicode()
    assert len(m) == 256
    assert len(set(m.values())) == 256
    assert m[ord("A")] == "A"
    assert m[ord(" ")] == "Ġ"


def test_merges_applied_in_rank_order() -> None:
    tok = _toy_tokenizer()
    ids = tok.encode(" the")
    assert [tok.decoder[i] for i in ids] == ["Ġthe"]
    ids = tok.encode("hello")
    # "he" (rank 1) merges before "ll" (rank 3); "ell" can no longer form.
    assert [tok.decoder[i] for i in ids] == ["he", "ll", "o"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "  leading and trailing spaces  ",
        "tabs\tand\nnewlines\r\n",
        "numbers 12345 and punctuation!?...",
        "unicode: héllo wörld, 你好, emoji 🤖🚀",
        "contractions: don't, I'll, we've",
    ],
)
def test_round_trip(text: str) -> None:
    tok = _toy_tokenizer()
    assert tok.decode(tok.encode(text)) == text


def test_special_token() -> None:
    tok = _toy_tokenizer()
    ids = tok.encode("a<|endoftext|>b")
    assert ids[1] == tok.eos_token_id
    assert tok.decode(ids) == "a<|endoftext|>b"
    assert tok.eos_token_id not in tok.encode("a<|endoftext|>b", allow_special=False)


def test_incremental_detokenizer_holds_partial_utf8() -> None:
    tok = _toy_tokenizer()
    ids = tok.encode("ok 🤖!")  # the robot emoji is 4 single-byte tokens here
    detok = IncrementalDetokenizer(tok)
    deltas = [detok.push([i]) for i in ids]
    assert "�" not in "".join(deltas)
    assert "".join(deltas) + detok.flush() == "ok 🤖!"
    assert "" in deltas  # bytes of the emoji were buffered until complete


@pytest.mark.slow
@pytest.mark.skipif(not is_downloaded("gpt2"), reason="needs GPT-2 tokenizer files")
def test_gpt2_tokenizer_known_ids() -> None:
    tok = load_tokenizer("gpt2")
    assert tok.encode("Hello world") == [15496, 995]
    assert tok.encode("<|endoftext|>") == [50256]
    text = "The quick brown fox 🦊 jumps over 13 lazy dogs.\n\nNew paragraph."
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.slow
@pytest.mark.skipif(not is_downloaded("gpt2"), reason="needs GPT-2 tokenizer files")
def test_gpt2_tokenizer_matches_huggingface() -> None:
    transformers = pytest.importorskip("transformers")
    from nanoserve.weights import model_dir

    ref = transformers.GPT2Tokenizer(
        str(model_dir("gpt2") / "vocab.json"), str(model_dir("gpt2") / "merges.txt")
    )
    tok = load_tokenizer("gpt2")
    texts = [
        "Hello, world! It's a beautiful day.",
        "def f(x):\n    return x ** 2  # square\n",
        "Ünïcödé   spacing\t\ttabs 3.14159 1,000,000",
        "😀 emoji and 漢字 and العربية",
    ]
    for t in texts:
        assert tok.encode(t) == ref.encode(t), t
