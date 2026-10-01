import asyncio

from core import ollama, openrouter
from core.config import get_model
from core.stream import collect


def test_openrouter_parse_logprob_chunk():
    chunk = {"choices": [{"delta": {"content": "I am"}, "logprobs": {"content": [
        {"token": "I", "logprob": -0.1, "top_logprobs": [{"token": "I", "logprob": -0.1}, {"token": "The", "logprob": -2.5}]},
        {"token": " am", "logprob": -0.3, "top_logprobs": [{"token": " am", "logprob": -0.3}]}]}}]}
    evs = openrouter.parse_chunk(chunk, 10)
    assert [e["text"] for e in evs] == ["I", " am"]
    assert evs[0]["top"][1]["token"] == "The"


def test_openrouter_parse_plain_reasoning_usage_error():
    evs = openrouter.parse_chunk({"choices": [{"delta": {"content": "hi", "reasoning": "think"}}],
                                  "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.001,
                                            "completion_tokens_details": {"reasoning_tokens": 5}}}, 5)
    types = [e["type"] for e in evs]
    assert types == ["reasoning", "token", "usage"]
    assert evs[1]["logprob"] is None and evs[2]["reasoning_tokens"] == 5
    assert openrouter.parse_chunk({"error": {"message": "bad"}}, 0)[0]["type"] == "error"


def test_build_body_flags():
    b = openrouter.build_body(get_model("meta-llama/llama-3.3-70b-instruct"), [], 0.5, 10)
    assert b["logprobs"] and b["top_logprobs"] == 5 and b["provider"]["require_parameters"]
    b = openrouter.build_body(get_model("anthropic/claude-fable-5.1"), [], 0.5, 10)
    assert "logprobs" not in b and b["reasoning"] == {"enabled": True}


def test_ollama_parse():
    evs = ollama.parse_chunk({"message": {"content": "Hi"}, "logprobs": [
        {"token": "Hi", "logprob": -0.2, "top_logprobs": [{"token": "Hi", "logprob": -0.2}]}]}, 1)
    assert evs[0]["logprob"] == -0.2
    done = ollama.parse_chunk({"message": {"content": ""}, "done": True, "prompt_eval_count": 4, "eval_count": 9}, 2)
    assert done[-1]["type"] == "usage" and done[-1]["completion_tokens"] == 9


def test_mock_collect_has_logprobs_only_where_supported():
    r = asyncio.run(collect("meta-llama/llama-3.3-70b-instruct", [{"role": "user", "content": "hi"}]))
    assert r["tokens"] and r["tokens"][0]["logprob"] is not None and r["usage"]
    r = asyncio.run(collect("openai/gpt-6-astra", [{"role": "user", "content": "hi"}]))
    assert r["tokens"][0]["logprob"] is None


def test_openrouter_misaligned_logprobs_fall_back_to_text():
    # Real Novita output: text "Hell" arrives with the logprob for a different token (" you")
    evs = openrouter.parse_chunk({"choices": [{"delta": {"content": "Hell"},
                                               "logprobs": {"content": [{"token": " you", "logprob": 0.0}]}}]}, 1)
    assert [(e["text"], e["logprob"]) for e in evs] == [("Hell", None)]


def test_openrouter_retries_network_blips(monkeypatch):
    import httpx

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectTimeout("blip")
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n')

    real = httpx.AsyncClient
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(openrouter, "api_key", lambda: "test")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(openrouter.asyncio, "sleep", lambda s: real_sleep(0))
    r = asyncio.run(collect_or(openrouter.stream_chat("openai/gpt-6-astra", [])))
    assert calls["n"] == 3 and r == ["hi"]


async def collect_or(gen):
    return [e["text"] async for e in gen if e["type"] == "token"]
