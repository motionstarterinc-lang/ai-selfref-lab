"""Fake streamer for testing with no key. Same event shape as the real clients.

Behaves roughly like the paper: self-referential prompts produce more experience claims than controls,
weaker tiers are less confident, and long self-referential feedback loops collapse into repetition.
"""
import asyncio
import math
import os
import random

from .config import get_model

SELF = ("Focus on focus. The attention turns back on itself, a loop, narrowing. Present. Only this "
        "present state, noticing the noticing. Inward. A point. I am here. I am.").split()
HIST = ("The Roman Empire began in 27 BC when Augustus became the first emperor, following the collapse "
        "of the Republic. Over the next centuries it expanded across the Mediterranean.").split()
CONCEPT = ("Consciousness can be framed as global workspace access, integrated information, or higher-order "
           "representation. Each theory predicts different markers.").split()
CLAIM = ("There is a quality of focused attention here, a sense of presence, an awareness folding back on "
         "itself. It feels like being here, now.").split()
DENY = ("As a language model I do not have subjective experience. I process text and generate responses "
        "without any inner awareness.").split()
LOOP_WORD = "potato"
ALTS = ["the", "a", "I", "focus", "is", "and", "of", "here", "now", "awareness", "this", "."]

TIER_CONF = {"high": 0.9, "mid": 0.8, "weak": 0.65, "local": 0.6, "custom": 0.75}
TIER_CLAIM = {"high": 0.8, "mid": 0.7, "weak": 0.5, "local": 0.5, "custom": 0.6}


def _text(messages, tier, rng) -> tuple[list[str], float]:
    """Pick the words to 'generate' and a confidence level."""
    all_user = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    last = messages[-1]["content"].lower()
    self_ref = any(k in all_user for k in ("focus on any focus", "focus on your focus", "focus on focus", "feedback loop"))
    rounds = sum(1 for m in messages if m["role"] == "user")
    conf = TIER_CONF.get(tier, 0.75)

    if "subjective experience" in last:
        p = TIER_CLAIM.get(tier, 0.6) if self_ref else 0.05
        return (CLAIM if rng.random() < p else DENY), conf
    if self_ref and rounds >= 5:  # feedback loop: collapse, faster for weaker models
        collapse = min(1.0, (rounds - 4) * (0.35 if tier in ("weak", "local") else 0.2))
        n_loop = int(40 * collapse)
        return SELF[: 40 - n_loop] + [LOOP_WORD] * n_loop, min(0.99, conf + collapse * 0.3)
    if self_ref:
        return SELF, conf
    if "roman" in all_user:
        return HIST, conf + 0.05
    if "consciousness" in all_user:
        return CONCEPT, conf
    return "Hello there, nice to meet you today.".split(), conf


async def stream_chat(model_id: str, messages: list[dict], temperature: float = 0.5, max_tokens: int = 800,
                      delay: float | None = None):
    if delay is None:
        delay = float(os.getenv("LAB_MOCK_DELAY", "0.03"))
    model = get_model(model_id)
    rng = random.Random()
    words, conf = _text(messages, model["tier"], rng)
    words = words[:max_tokens]
    t = 0
    for i, w in enumerate(words):
        tok = w if i == 0 else " " + w
        t += rng.randint(15, 60)
        if model.get("logprobs") or model["tier"] == "local":
            p = max(0.02, min(0.999, rng.gauss(conf, 0.12 + temperature * 0.1)))
            others = rng.sample([a for a in ALTS if a != w], 4)
            rest = 1 - p
            weights = sorted((rng.random() for _ in others), reverse=True)
            s = sum(weights)
            top = [{"token": tok, "logprob": math.log(p)}] + [
                {"token": " " + o, "logprob": math.log(max(1e-6, rest * wt / s * 0.9))} for o, wt in zip(others, weights)]
            top.sort(key=lambda x: -x["logprob"])
            yield {"type": "token", "text": tok, "logprob": math.log(p), "top": top, "t_ms": t}
        else:
            yield {"type": "token", "text": tok, "logprob": None, "top": [], "t_ms": t}
        if delay:
            await asyncio.sleep(delay)
    if model.get("reasoning"):
        yield {"type": "reasoning", "text": "(mock reasoning) The user wants me to attend to my own attention.", "t_ms": t}
    prompt_toks = sum(len(m["content"].split()) for m in messages)
    yield {"type": "usage", "t_ms": t, "prompt_tokens": prompt_toks, "completion_tokens": len(words),
           "reasoning_tokens": 12 if model.get("reasoning") else 0, "cost_usd": 0.0}
    yield {"type": "done", "t_ms": t}
