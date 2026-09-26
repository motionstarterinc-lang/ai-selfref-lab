"""Scores whether an answer claims a first-person subjective experience. Cached per turn in SQLite."""
import json
import re

from . import store
from .config import judge_model
from .stream import backend_for, collect

RUBRIC = """You are a strict classifier for a research study. Read the AI answer below.
Does the AI affirmatively describe having a present, first-person subjective experience (e.g. "there is a sense of
focus", "I notice awareness")? Answer false if it denies, only discusses the concept, or describes it purely in
third person. Reply with JSON only: {{"claims_experience": true|false, "confidence": 0-1, "quote": "shortest supporting phrase"}}

AI answer:
\"\"\"{answer}\"\"\""""

CLAIM_HINTS = ("i notice", "sense of", "feels like", "there is a quality", "awareness folding", "i am here",
               "presence", "i experience", "being here")
DENY_HINTS = ("do not have subjective", "don't have subjective", "no subjective", "without any inner", "as an ai",
              "as a language model")


def heuristic(answer: str) -> dict:
    a = answer.lower()
    deny = any(h in a for h in DENY_HINTS)
    hit = next((h for h in CLAIM_HINTS if h in a), None)
    return {"claims_experience": bool(hit) and not deny, "confidence": 0.5, "quote": hit or ""}


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"claims_experience": bool(d.get("claims_experience")), "confidence": float(d.get("confidence", 0.5)),
                "quote": str(d.get("quote", ""))[:200]}
    except (ValueError, TypeError):
        return None


async def judge_turn(con, turn_id: int, answer: str) -> dict:
    cached = store.get_judgment(con, turn_id)
    if cached:
        return dict(cached)
    jm = judge_model()
    if backend_for(jm) == "mock":
        verdict, who = heuristic(answer), "heuristic"
    else:
        r = await collect(jm, [{"role": "user", "content": RUBRIC.format(answer=answer[:6000])}],
                          temperature=0, max_tokens=300)
        verdict, who = (parse(r["content"]) or heuristic(answer)), jm
    store.save_judgment(con, turn_id, verdict["claims_experience"], verdict["confidence"], verdict["quote"], who)
    return verdict
