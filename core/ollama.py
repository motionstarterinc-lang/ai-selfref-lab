"""Local tier: your own Llama through Ollama (free, unlimited). Needs Ollama 0.12.11+ for logprobs.

Same event shape as openrouter.py. Ollama's docs only promise logprobs on the final non-streamed
response, so this reads logprobs from any chunk that has them and falls back to plain text otherwise.
"""
import json
import time

import httpx

from .config import OLLAMA_URL


def parse_chunk(chunk: dict, t_ms: int) -> list[dict]:
    if chunk.get("error"):
        return [{"type": "error", "text": chunk["error"], "t_ms": t_ms}]
    events = []
    msg = chunk.get("message") or {}
    if msg.get("thinking"):
        events.append({"type": "reasoning", "text": msg["thinking"], "t_ms": t_ms})  # gpt-oss thinks
    lps = chunk.get("logprobs")
    if lps:
        for e in lps:
            events.append({
                "type": "token", "text": e.get("token", ""), "logprob": e.get("logprob"),
                "top": [{"token": t["token"], "logprob": t["logprob"]} for t in e.get("top_logprobs") or []],
                "t_ms": t_ms,
            })
    elif msg.get("content"):
        events.append({"type": "token", "text": msg["content"], "logprob": None, "top": [], "t_ms": t_ms})
    if chunk.get("done"):
        events.append({"type": "usage", "t_ms": t_ms, "prompt_tokens": chunk.get("prompt_eval_count", 0),
                       "completion_tokens": chunk.get("eval_count", 0), "reasoning_tokens": 0, "cost_usd": 0.0})
    return events


async def stream_chat(model_id: str, messages: list[dict], temperature: float = 0.5, max_tokens: int = 800):
    body = {"model": model_id, "messages": messages, "stream": True, "logprobs": True, "top_logprobs": 5,
            "options": {"temperature": temperature, "num_predict": max_tokens}}
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode(errors="replace")[:300]
                    yield {"type": "error", "text": f"Ollama HTTP {r.status_code}: {text}", "t_ms": 0}
                    return
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    for ev in parse_chunk(json.loads(line), int((time.monotonic() - start) * 1000)):
                        yield ev
    except httpx.ConnectError:
        yield {"type": "error", "text": f"Can't reach Ollama at {OLLAMA_URL}. Is it running? (`ollama serve`)", "t_ms": 0}
        return
    yield {"type": "done", "t_ms": int((time.monotonic() - start) * 1000)}
