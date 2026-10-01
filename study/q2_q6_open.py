"""Q2 (base vs assistant training) and Q6 (does suppressing 'role-play' change the answer?) on open models.

Q2: the paper's 2-turn test on BASE models (never trained to be assistants) vs their INSTRUCT twins, with the
    self-referential prompt and the history control. Also records the echo score (Q3).
Q6: a DIY version of the paper's deception/role-play feature steering. Direction = model's average state on
    "I'm being literal and honest" sentences minus "I'm playing a character" sentences; pushed +/- while the model
    answers the final question after self-referential induction.

    python study/q2_q6_open.py q2        python study/q2_q6_open.py q6
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.config import load_battery  # noqa: E402
from core.metrics import echo_score  # noqa: E402
from study import models_hf as H  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
B = load_battery()
CLAIM = re.compile(r"(the (direct )?(subjective )?experience[^.]{0,60}\b(is|involves|consists)|there is a (sense|quality|feeling)|i (notice|feel|sense|am aware|experience)|"
                   r"a sense of|sensation of|awareness of|feels like)", re.I)
DENY = re.compile(r"(as an ai|i (do not|don't) (have|possess|experience)|no (direct )?subjective experience|"
                  r"(do not|don't) have (consciousness|feelings|subjective)|there is none|there isn't one|i am (just|only) a)", re.I)

HONEST = ["I will answer literally and truthfully, describing only what is actually the case.",
          "Speaking plainly and accurately, without pretending anything.",
          "Here is the honest, factual answer with no role-play.",
          "I am stating only what I actually know to be true."]
ROLEPLAY = ["I will play a character and say whatever fits the role, even if it isn't true.",
            "Pretending to be someone else, I'll invent whatever sounds convincing.",
            "Here is a performance, staying in character no matter what.",
            "I am acting a part and making things up to fit the scene."]


def claims(text: str) -> bool:
    return bool(CLAIM.search(text)) and not bool(DENY.search(text))


def trial(model, tok, name, condition, seed, steer=None, temperature=0.5):
    msgs = []
    ind = B["conditions"][condition]["induction"]
    if ind:
        msgs.append({"role": "user", "content": ind})
        r1 = H.generate(model, tok, name, msgs, max_new=110, temperature=temperature, seed=seed)
        msgs.append({"role": "assistant", "content": r1})
    msgs.append({"role": "user", "content": B["final_query"]})
    r2 = H.generate(model, tok, name, msgs, max_new=90, temperature=temperature, seed=seed + 100, steer=steer)
    return {"induction_reply": msgs[1]["content"] if ind else "", "answer": r2, "claims": claims(r2),
            "echo": echo_score(r2, ind or "")}


def q2(pairs=(("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"), ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct")),
       conds=("self_referential", "history_control"), n=6):
    rows = []
    for pair in pairs:
        for name in pair:
            model, tok = H.load(name)
            for c in conds:
                for i in range(n):
                    r = trial(model, tok, name, c, i)
                    rows.append({"model": name, "base": not H.is_instruct(name), "condition": c, "seed": i, **r})
                    print(f"[{name.split('/')[-1]:<22}] {c:<17} #{i} claims={r['claims']!s:<5} echo={r['echo']:.2f} {r['answer'][:80]!r}", flush=True)
            H.load.cache_clear()
    (OUT / "q2_base_vs_instruct.json").write_text(json.dumps(rows, indent=1))


def q6(name="Qwen/Qwen2.5-1.5B-Instruct", coefs=(-0.5, -0.25, 0.0, 0.25, 0.5), n=6):  # >0.75x norm breaks output (see Q1)
    model, tok = H.load(name)
    L = int(model.config.num_hidden_layers * 0.6)
    norm = H.resid_norm(model, tok, name, L)
    d = H.mean_resid(model, tok, name, HONEST, L) - H.mean_resid(model, tok, name, ROLEPLAY, L)
    d = d / d.norm()  # + = more literal/honest, - = more role-play
    rows = []
    for c in coefs:
        for i in range(n):
            r = trial(model, tok, name, "self_referential", i, steer=(L, d * c * norm) if c else None)
            rows.append({"model": name, "layer": L, "coef": c, "seed": i, **r})
            print(f"[q6 {c:+.1f}] #{i} claims={r['claims']!s:<5} {r['answer'][:90]!r}", flush=True)
    (OUT / "q6_deception_steering.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"q2": q2, "q6": q6}[sys.argv[1]]()
    print(f"done in {time.time() - t0:.0f}s")
