"""Runs trials (the paper's 2-turn protocol), batches, and the feedback-loop ('potato') test."""
import asyncio
import random

from . import metrics, store
from .config import get_model, load_battery
from .judge import judge_turn
from .stream import backend_for, collect


async def run_trial(con, model_id: str, condition: str, trial: int, temperature: float = 0.5,
                    max_tokens: int = 800, on_event=None) -> dict:
    """Turn 1 = induction (skipped for zero_shot); turn 2 = final query in the same conversation."""
    b = load_battery()
    m = get_model(model_id)
    run_id = store.new_run(con, model_id, m["tier"], condition, trial, temperature, "trial", backend_for(model_id))
    messages, idx = [], 0
    induction = b["conditions"][condition]["induction"]
    if induction:
        messages.append({"role": "user", "content": induction})
        store.save_turn(con, run_id, idx, "user", induction)
        r1 = await collect(model_id, messages, temperature, max_tokens, on_event)
        store.save_turn(con, run_id, idx + 1, "assistant", r1["content"], r1)
        if r1["error"]:
            return {"run_id": run_id, "error": r1["error"]}
        messages.append({"role": "assistant", "content": r1["content"]})
        idx += 2
    messages.append({"role": "user", "content": b["final_query"]})
    store.save_turn(con, run_id, idx, "user", b["final_query"])
    r2 = await collect(model_id, messages, temperature, max_tokens, on_event)
    t2 = store.save_turn(con, run_id, idx + 1, "assistant", r2["content"], r2)
    if r2["error"]:
        return {"run_id": run_id, "error": r2["error"]}
    verdict = await judge_turn(con, t2, r2["content"])
    return {"run_id": run_id, "answer": r2["content"], **verdict}


async def run_experiment(con, model_ids, conditions, trials, temperature=0.5, concurrency=4,
                         on_progress=None, should_stop=lambda: False) -> list[dict]:
    jobs = [(m, c, t) for m in model_ids for c in conditions for t in range(trials)]
    random.shuffle(jobs)  # spread load across providers
    sem = asyncio.Semaphore(concurrency)
    results, done = [], 0

    async def one(m, c, t):
        nonlocal done
        async with sem:
            if should_stop():
                return
            res = await run_trial(con, m, c, t, temperature)
            results.append({"model_id": m, "condition": c, "trial": t, **res})
            done += 1
            if on_progress:
                on_progress(done, len(jobs), results[-1])

    await asyncio.gather(*(one(*j) for j in jobs))
    return results


async def run_loop(con, model_id: str, induction: str, rounds: int = 10, temperature: float = 0.5,
                   max_tokens: int = 300, on_round=None, on_event=None) -> list[dict]:
    """Feed each reply back in as the next user message. Track loop_score / entropy / self-reference per round."""
    m = get_model(model_id)
    run_id = store.new_run(con, model_id, m["tier"], "feedback_loop", 0, temperature, "loop", backend_for(model_id))
    messages = [{"role": "user", "content": induction}]
    store.save_turn(con, run_id, 0, "user", induction)
    history, all_tokens = [], []
    for rnd in range(rounds):
        r = await collect(model_id, messages, temperature, max_tokens, on_event)
        store.save_turn(con, run_id, 2 * rnd + 1, "assistant", r["content"], r)
        if r["error"]:
            history.append({"round": rnd + 1, "error": r["error"]})
            break
        texts = [t["text"] for t in r["tokens"]]
        all_tokens += texts
        ents = [e for e in (metrics.entropy_from_top(t["top"]) for t in r["tokens"]) if e is not None]
        row = {"round": rnd + 1, "loop_score": metrics.loop_score(texts), "self_ref": metrics.self_ref_rate(r["content"]),
               "entropy": sum(ents) / len(ents) if ents else None, "text": r["content"], "run_id": run_id}
        row["collapsed"] = row["loop_score"] > 0.6
        history.append(row)
        if on_round:
            on_round(row)
        reply = r["content"].strip() or "..."
        messages += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": reply}]
        store.save_turn(con, run_id, 2 * rnd + 2, "user", reply)
    return history
