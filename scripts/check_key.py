"""Phase 0 check: is the OpenRouter key set, and how much credit is left?"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import api_key  # noqa: E402
from core.openrouter import key_info  # noqa: E402

if not api_key():
    print("No OPENROUTER_API_KEY set. Copy .env.example to .env and paste your key. (Mock mode works without it.)")
    sys.exit(0)
info = asyncio.run(key_info())
limit = info.get("limit")
print(f"Key OK. Used: ${info.get('usage', 0):.4f}  Limit: {'none' if limit is None else f'${limit}'}  "
      f"Remaining: {info.get('limit_remaining', 'n/a')}")
