"""Live prices from OpenRouter's public models API, cached for a day."""
import json
import time

import httpx

from .config import OPENROUTER_URL, ROOT, get_model

CACHE = ROOT / "data" / "pricing.json"
# Per-trial token assumptions (2 turns: induction + final query). Reasoning models also burn hidden tokens.
IN_TOKENS, OUT_TOKENS, REASONING_EXTRA = 900, 1000, 1000


def prices() -> dict:
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 86400:
        return json.loads(CACHE.read_text())
    try:
        data = httpx.get(f"{OPENROUTER_URL}/models", timeout=20).json()["data"]
        table = {m["id"]: {"in": float(m["pricing"]["prompt"]), "out": float(m["pricing"]["completion"])}
                 for m in data if float(m["pricing"].get("prompt", -1)) >= 0}
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(table))
        return table
    except Exception:
        return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def estimate(model_id: str, trials: int, rounds: int = 1) -> float | None:
    """USD estimate for `trials` two-turn trials. 0 for local, None if price unknown."""
    m = get_model(model_id)
    if m["provider"] == "ollama":
        return 0.0
    p = prices().get(model_id)
    if not p:
        return None
    out = OUT_TOKENS + (REASONING_EXTRA if m.get("reasoning") else 0)
    return trials * rounds * (IN_TOKENS * p["in"] + out * p["out"])
