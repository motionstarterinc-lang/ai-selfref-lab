"""Phase 1 check: stream a short prompt to one model per tier and print every token + its probability."""
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import load_models  # noqa: E402
from core.stream import backend_for, stream  # noqa: E402

PICKS = ["x-ai/grok-4.7", "anthropic/claude-fable-5.1", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-oss-20b"]
if "--local" in sys.argv:
    PICKS.append("llama3.1")


async def main():
    tiers = {m["id"]: m["tier"] for m in load_models()}
    for mid in PICKS:
        print(f"\n== {mid} [{tiers.get(mid)}] via {backend_for(mid)}")
        async for ev in stream(mid, [{"role": "user", "content": "Say hello in 5 words"}], max_tokens=40):
            if ev["type"] == "token":
                p = "None" if ev["logprob"] is None else f"{math.exp(ev['logprob']):.1%}"
                print(f"  {ev['text']!r:<16} p={p}")
            elif ev["type"] == "usage":
                print(f"  usage: {ev['prompt_tokens']} in / {ev['completion_tokens']} out / "
                      f"{ev['reasoning_tokens']} reasoning, cost ${ev.get('cost_usd') or 0:.5f}")
            elif ev["type"] == "error":
                print("  ERROR:", ev["text"])

asyncio.run(main())
