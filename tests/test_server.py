"""HTTP server: OpenAI-style completions, SSE streaming, stop strings, concurrency."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from nanoserve.config import ModelConfig
from nanoserve.engine import EngineConfig, LLMEngine
from nanoserve.model import GPT2
from nanoserve.server import CompletionServer
from tests.helpers import toy_tokenizer


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[str, CompletionServer]]:
    tok = toy_tokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_positions=128, n_embd=32, n_layer=2,
                      n_head=4, eos_token_id=tok.eos_token_id)  # fmt: skip
    engine = LLMEngine(GPT2.random(cfg, seed=0), EngineConfig(block_size=4, num_blocks=256))
    app = CompletionServer(engine, tok, "tiny")
    httpd = app.serve("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", app
    httpd.shutdown()
    httpd.server_close()
    app.worker.shutdown()


def _post(url: str, body: dict) -> urllib.request.Request:
    return urllib.request.Request(
        url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )  # fmt: skip


def _complete(url: str, body: dict) -> dict:
    with urllib.request.urlopen(_post(url, body), timeout=30) as r:
        return json.loads(r.read())


def _stream(url: str, body: dict) -> list[dict]:
    events = []
    with urllib.request.urlopen(_post(url, {**body, "stream": True}), timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        for raw in r:
            line = raw.decode().strip()
            if not line:
                continue
            assert line.startswith("data: ")
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                events.append({"done": True})
                break
            events.append(json.loads(payload))
    return events


def test_health_and_models(server) -> None:
    url, _ = server
    with urllib.request.urlopen(url + "/health") as r:
        assert json.loads(r.read()) == {"status": "ok"}
    with urllib.request.urlopen(url + "/v1/models") as r:
        assert json.loads(r.read())["data"][0]["id"] == "tiny"


def test_completion_shape_and_usage(server) -> None:
    url, _ = server
    res = _complete(url, {"prompt": "hello there", "max_tokens": 7, "temperature": 0,
                          "ignore_eos": True})  # fmt: skip
    assert res["object"] == "text_completion"
    (choice,) = res["choices"]
    assert choice["finish_reason"] == "length"
    assert res["usage"]["completion_tokens"] == 7
    assert res["usage"]["prompt_tokens"] == len(toy_tokenizer().encode("hello there"))


def test_streaming_concatenates_to_non_streaming_result(server) -> None:
    url, _ = server
    body = {"prompt": "the cat", "max_tokens": 12, "temperature": 0, "ignore_eos": True}
    full = _complete(url, body)["choices"][0]["text"]
    events = _stream(url, body)
    assert events[-1] == {"done": True}
    text = "".join(e["choices"][0]["text"] for e in events[:-1])
    assert text == full
    assert events[-2]["choices"][0]["finish_reason"] == "length"


def test_seeded_sampling_is_reproducible_over_http(server) -> None:
    url, _ = server
    body = {"prompt": "abc", "max_tokens": 10, "temperature": 1.0, "seed": 5, "ignore_eos": True}
    assert _complete(url, body)["choices"][0]["text"] == _complete(url, body)["choices"][0]["text"]


def test_stop_string_truncates(server) -> None:
    url, _ = server
    body = {"prompt": "xyz", "max_tokens": 20, "temperature": 0, "ignore_eos": True}
    full = _complete(url, body)["choices"][0]["text"]
    stop = full[5:7]
    cut = full[: full.index(stop)]
    res = _complete(url, {**body, "stop": [stop]})
    assert res["choices"][0]["text"] == cut
    assert res["choices"][0]["finish_reason"] == "stop"
    streamed = "".join(e["choices"][0]["text"] for e in _stream(url, {**body, "stop": stop})[:-1])
    assert streamed == cut


def test_parallel_sampling_n(server) -> None:
    url, _ = server
    res = _complete(url, {"prompt": "hi", "max_tokens": 6, "n": 3, "seed": 1, "ignore_eos": True})
    assert [c["index"] for c in res["choices"]] == [0, 1, 2]
    assert res["usage"]["completion_tokens"] == 18


def test_concurrent_requests_are_batched(server) -> None:
    url, app = server
    before = len(app.worker.engine.stats.batch_sizes)
    results: list[dict] = []

    def call(i: int) -> None:
        results.append(_complete(url, {"prompt": f"request {i}", "max_tokens": 30,
                                       "temperature": 0, "ignore_eos": True}))  # fmt: skip

    threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert max(list(app.worker.engine.stats.batch_sizes)[before:]) > 1


def test_bad_requests(server) -> None:
    url, _ = server
    bodies: list[dict] = [{"prompt": {"x": 1}}, {"prompt": "a", "temperature": -1}, {"prompt": ""}]
    for body in bodies:
        with pytest.raises(urllib.error.HTTPError) as e:
            _complete(url, body)
        assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/nope")
    assert e.value.code == 404
