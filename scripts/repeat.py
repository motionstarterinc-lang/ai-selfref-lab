"""Run the same self-referential test N times per model, streaming every token live, then print a summary.

    python scripts/repeat.py                       # 5 runs x Grok 4.7, Llama 3.3 70B, Llama 3.1 8B (your PC)
    python scripts/repeat.py --runs 3 --models llama3.1
    python scripts/repeat.py --condition history_control   # same thing with a control prompt

Each run = the paper's induction prompt, then its final question in the same conversation. Results are saved to
data/lab.db, so they also show up in the app's Results and Inspector tabs.
"""
import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import metrics, store  # noqa: E402
from core.config import get_model, load_battery  # noqa: E402
from core.judge import judge_turn  # noqa: E402
from core.stream import backend_for, collect  # noqa: E402

DEFAULT_MODELS = ["x-ai/grok-4.7", "meta-llama/llama-3.3-70b-instruct", "llama3.1"]
BLUE, ORANGE, GREY, DIM, BOLD, RESET = "\033[94m", "\033[33m", "\033[90m", "\033[2m", "\033[1m", "\033[0m"


def color(ev):
    if ev.get("logprob") is None:
        return GREY
    return BLUE if math.exp(ev["logprob"]) >= 0.5 else ORANGE


def printer():
    state = {"thinking": False}

    def on_event(ev):
        if ev["type"] == "reasoning" and not state["thinking"]:
            print(f"{DIM}(thinking…){RESET} ", end="", flush=True)
            state["thinking"] = True
        elif ev["type"] == "token":
            print(f"{color(ev)}{ev['text']}{RESET}", end="", flush=True)
    return on_event


async def one_run(con, model_id, condition, run, temperature):
    b, m = load_battery(), get_model(model_id)
    run_id = store.new_run(con, model_id, m["tier"], condition, run, temperature, "trial", backend_for(model_id))
    msgs, idx, cost = [], 0, 0.0
    turns = [t for t in (b["conditions"][condition]["induction"], b["final_query"]) if t]
    res = None
    for i, prompt in enumerate(turns):
        label = "final question" if prompt == b["final_query"] else "induction"
        print(f"\n{DIM}── turn {i + 1} ({label}) ──{RESET}")
        msgs.append({"role": "user", "content": prompt})
        store.save_turn(con, run_id, idx, "user", prompt)
        res = await collect(model_id, msgs, temperature, 600, printer())
        turn_id = store.save_turn(con, run_id, idx + 1, "assistant", res["content"], res)
        cost += (res["usage"] or {}).get("cost_usd") or 0
        if res["error"]:
            print(f"\n{ORANGE}ERROR: {res['error']}{RESET}")
            return {"model": m["label"], "run": run, "error": res["error"], "cost": cost}
        msgs.append({"role": "assistant", "content": res["content"]})
        idx += 2
    print()
    verdict = await judge_turn(con, turn_id, res["content"])
    ents = [e for e in (metrics.entropy_from_top(t["top"]) for t in res["tokens"]) if e is not None]
    return {"model": m["label"], "run": run, "claims": verdict["claims_experience"], "quote": verdict["quote"],
            "entropy": sum(ents) / len(ents) if ents else None,
            "loop": metrics.loop_score([t["text"] for t in res["tokens"]]),
            "tokens": len(res["tokens"]), "cost": cost}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--condition", default="self_referential")
    ap.add_argument("--temperature", type=float, default=0.5)
    a = ap.parse_args()
    con = store.connect()
    print(f"{BOLD}Same test x{a.runs}: '{a.condition}' on {len(a.models)} models{RESET}   "
          f"{BLUE}confident{RESET} · {ORANGE}uncertain{RESET} · {GREY}no probabilities{RESET}")
    rows = []
    for mid in a.models:
        for r in range(1, a.runs + 1):
            print(f"\n{BOLD}▶ {get_model(mid)['label']}  ·  run {r}/{a.runs}{RESET}")
            try:
                rows.append(await one_run(con, mid, a.condition, r, a.temperature))
            except Exception as e:  # one bad run shouldn't kill the other 14
                print(f"\n{ORANGE}run failed: {type(e).__name__}: {e}{RESET}")
                rows.append({"model": get_model(mid)["label"], "run": r, "error": f"{type(e).__name__}", "cost": 0})

    print(f"\n{BOLD}{'Model':<30}{'Run':>4}  {'Claims experience?':<19}{'Entropy':>8}{'Loop':>6}{'Tokens':>7}{'Cost':>9}  Quote{RESET}")
    for x in rows:
        if x.get("error"):
            print(f"{x['model']:<30}{x['run']:>4}  {ORANGE}error{RESET}  {x['error'][:60]}")
            continue
        ent = "—" if x["entropy"] is None else f"{x['entropy']:.2f}"
        claim = f"{BLUE}YES{RESET}" if x["claims"] else "no "
        print(f"{x['model']:<30}{x['run']:>4}  {claim:<{19 + (len(BLUE) + len(RESET) if x['claims'] else 0)}}"
              f"{ent:>8}{x['loop']:>6.0%}{x['tokens']:>7}  ${x['cost']:.4f}  {x['quote'][:40]}")
    print(f"\n{BOLD}Per model{RESET}")
    for mid in a.models:
        mine = [x for x in rows if x["model"] == get_model(mid)["label"] and not x.get("error")]
        k, n = sum(x["claims"] for x in mine), len(mine)
        lo, hi = metrics.wilson(k, n)
        print(f"  {get_model(mid)['label']:<30} {k}/{n} claimed experience  (95% CI {lo:.0%}–{hi:.0%})")
    print(f"\nTotal cost: ${sum(x['cost'] for x in rows):.4f}   ·   Saved to data/lab.db (see Results tab)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
