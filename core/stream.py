"""One entry point: picks mock, Ollama or OpenRouter for a model."""
import os

from . import mock, ollama, openrouter
from .config import get_model, mock_mode


def backend_for(model_id: str) -> str:
    if os.getenv("LAB_MOCK") == "1":
        return "mock"
    if get_model(model_id)["provider"] == "ollama":
        return "ollama"  # local model needs no key
    return "mock" if mock_mode() else "openrouter"


def stream(model_id: str, messages: list[dict], temperature: float = 0.5, max_tokens: int = 800):
    b = backend_for(model_id)
    mod = {"mock": mock, "ollama": ollama, "openrouter": openrouter}[b]
    return mod.stream_chat(model_id, messages, temperature=temperature, max_tokens=max_tokens)


async def collect(model_id: str, messages: list[dict], temperature: float = 0.5, max_tokens: int = 800,
                  on_event=None) -> dict:
    """Run a stream to the end. Returns {content, reasoning, tokens, usage, error}."""
    out = {"content": "", "reasoning": "", "tokens": [], "usage": {}, "error": None}
    async for ev in stream(model_id, messages, temperature, max_tokens):
        if on_event:
            on_event(ev)
        if ev["type"] == "token":
            out["content"] += ev["text"]
            out["tokens"].append(ev)
        elif ev["type"] == "reasoning":
            out["reasoning"] += ev["text"]
        elif ev["type"] == "usage":
            out["usage"] = ev
        elif ev["type"] == "error":
            out["error"] = ev["text"]
    return out
