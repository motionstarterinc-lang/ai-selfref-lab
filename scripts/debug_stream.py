"""Print the raw OpenRouter stream for one model (first N data lines), to debug token parsing.
Never prints your key.   python scripts/debug_stream.py x-ai/grok-4.7 [--no-reasoning]"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import OPENROUTER_URL, api_key, get_model  # noqa: E402
from core.openrouter import build_body  # noqa: E402

model_id = sys.argv[1]
m = dict(get_model(model_id))
if "--no-reasoning" in sys.argv:
    m["reasoning"] = False
body = build_body(m, [{"role": "user", "content": "Say hello in 5 words"}], 0.5, 60)
print("request flags:", {k: body[k] for k in body if k not in ("messages", "model")})
with httpx.stream("POST", f"{OPENROUTER_URL}/chat/completions", json=body, timeout=60,
                  headers={"Authorization": f"Bearer {api_key()}"}) as r:
    print("HTTP", r.status_code)
    n = 0
    for line in r.iter_lines():
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[6:])
        ch = (d.get("choices") or [{}])[0]
        lp = (ch.get("logprobs") or {}).get("content")
        print(f"[{n}] provider={d.get('provider')} content={ch.get('delta', {}).get('content')!r} "
              f"reasoning={bool(ch.get('delta', {}).get('reasoning'))} "
              f"logprobs={[(e.get('token'), round(e.get('logprob', 0), 2)) for e in lp] if lp else lp!r}")
        n += 1
        if n >= 25:
            break
