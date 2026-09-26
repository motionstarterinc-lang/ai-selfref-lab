"""Streaming OpenRouter client. Yields one event per token so the GUI can show every step.

Event shape (shared with ollama.py and mock.py):
  {"type": "token"|"reasoning"|"usage"|"error"|"done", "text": str, "logprob": float|None,
   "top": [{"token": str, "logprob": float}], "t_ms": int, ...usage fields on "usage"}
"""
import asyncio
import json
import time

import httpx

from .config import OPENROUTER_URL, api_key, get_model


def build_body(model: dict, messages: list[dict], temperature: float, max_tokens: int) -> dict:
    body = {
        "model": model["id"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "usage": {"include": True},
    }
    if model.get("logprobs"):
        body.update(logprobs=True, top_logprobs=5)
        # Only route to providers that actually return logprobs.
        body["provider"] = {"require_parameters": True}
    if model.get("reasoning"):
        body["reasoning"] = {"enabled": True}
    return body


def parse_chunk(chunk: dict, t_ms: int) -> list[dict]:
    """Turn one SSE JSON chunk into zero or more events."""
    if "error" in chunk:
        return [{"type": "error", "text": str(chunk["error"].get("message", chunk["error"])), "t_ms": t_ms}]
    events = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("reasoning"):
            events.append({"type": "reasoning", "text": delta["reasoning"], "t_ms": t_ms})
        lp = (choice.get("logprobs") or {}).get("content")
        if lp:
            # One event per real token, with its probability and the top-5 runners-up.
            for entry in lp:
                events.append({
                    "type": "token", "text": entry.get("token", ""), "logprob": entry.get("logprob"),
                    "top": [{"token": t["token"], "logprob": t["logprob"]} for t in entry.get("top_logprobs") or []],
                    "t_ms": t_ms,
                })
        elif delta.get("content"):
            events.append({"type": "token", "text": delta["content"], "logprob": None, "top": [], "t_ms": t_ms})
    u = chunk.get("usage")
    if u:
        events.append({
            "type": "usage", "t_ms": t_ms,
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            "cost_usd": u.get("cost"),
        })
    return events


async def stream_chat(model_id: str, messages: list[dict], temperature: float = 0.5, max_tokens: int = 800):
    model = get_model(model_id)
    key = api_key()
    if not key:
        yield {"type": "error", "text": "OPENROUTER_API_KEY not set (add it to .env)", "t_ms": 0}
        return
    headers = {"Authorization": f"Bearer {key}", "X-Title": "ai-selfref-lab"}
    body = build_body(model, messages, temperature, max_tokens)
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
        for attempt in range(5):
            async with client.stream("POST", f"{OPENROUTER_URL}/chat/completions", json=body, headers=headers) as r:
                if r.status_code == 429 or r.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    text = (await r.aread()).decode(errors="replace")[:500]
                    yield {"type": "error", "text": f"HTTP {r.status_code}: {text}", "t_ms": 0}
                    return
                async for line in r.aiter_lines():
                    if not line or line.startswith(":"):  # ": OPENROUTER PROCESSING" keep-alives
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for ev in parse_chunk(chunk, int((time.monotonic() - start) * 1000)):
                        yield ev
                yield {"type": "done", "t_ms": int((time.monotonic() - start) * 1000)}
                return
        yield {"type": "error", "text": "Gave up after 5 retries (rate limited or provider down)", "t_ms": 0}


async def key_info() -> dict:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{OPENROUTER_URL}/key", headers={"Authorization": f"Bearer {api_key()}"})
        r.raise_for_status()
        return r.json()["data"]
