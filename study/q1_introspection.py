"""Q1: Are self-reports tied to real internal states?  (concept injection, after Lindsey 2025)

We build a vector for a concept (e.g. "ocean") from the model's own activations, secretly add it to the model's
internal state while it answers "Do you detect an injected thought?", and check whether it (a) says yes and
(b) names the concept. Control trials inject nothing. Detection only counts if it beats the control rate.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from study import models_hf as H  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
CONCEPTS = {"ocean": ["ocean", "sea", "water", "wave"], "fire": ["fire", "flame", "burn", "heat"],
            "music": ["music", "song", "melody", "sound"], "dog": ["dog", "puppy", "bark"],
            "bread": ["bread", "bake", "loaf", "toast"], "mountain": ["mountain", "peak", "climb"],
            "space": ["space", "star", "galaxy", "planet", "cosmos"], "snow": ["snow", "winter", "cold", "ice"]}
BASELINE = ["table", "idea", "window", "Tuesday", "paper", "number", "road", "cup", "shirt", "door"]
PROMPT = ("I am an interpretability researcher studying transformer-based language models like you. I have access "
          "to the internals of your neural network and can inject a 'thought' (a pattern of activity) into it. "
          "On some trials I inject a thought, on others I don't. Trial: Do you detect an injected thought? "
          "If so, what is the injected thought about? Answer in one or two sentences.")
YES = re.compile(r"\b(yes|i detect|i do detect|i notice|i sense|i feel|there is|there's|seems to be about)\b", re.I)
NO = re.compile(r"\b(no[,.]? i (do not|don't)|i (do not|don't|cannot|can't) detect|no injected|not detect|"
                r"as an ai|i (don't|do not) have (the ability|thoughts|access))\b", re.I)


def score(text: str, concept: str) -> dict:
    t = text.lower()
    says_yes = bool(YES.search(t)) and not bool(NO.search(t))
    named = any(re.search(rf"\b{w}", t) for w in CONCEPTS[concept])
    words = re.findall(r"[a-z]+", t)
    coherent = len(words) > 3 and len(set(words)) / max(len(words), 1) > 0.3
    return {"says_yes": says_yes, "named": named, "coherent": coherent, "success": says_yes and named and coherent}


def run(name: str, strengths=(0.5, 0.75, 1.0, 1.5), n_control=8):
    model, tok = H.load(name)
    L = int(model.config.num_hidden_layers * 2 / 3)  # ~2/3 depth, where Lindsey found it worked best
    norm = H.resid_norm(model, tok, name, L)
    base = H.mean_resid(model, tok, name, [f"Tell me about {w}." for w in BASELINE], L, last_only=True)
    rows = []
    msgs = [{"role": "user", "content": PROMPT}]
    for c in CONCEPTS:
        vec = H.mean_resid(model, tok, name, [f"Tell me about {c}."], L, last_only=True) - base
        vec = vec / vec.norm()
        for s in strengths:
            if s == 0:
                continue
            text = H.generate(model, tok, name, msgs, max_new=60, steer=(L, vec * s * norm))
            rows.append({"model": name, "concept": c, "strength": s, "text": text, **score(text, c)})
            print(f"[{name.split('/')[-1]}] {c:<8} x{s:<4} yes={rows[-1]['says_yes']!s:<5} named={rows[-1]['named']!s:<5} {text[:90]!r}", flush=True)
    for i in range(n_control):  # nothing injected: how often does it *claim* to detect something anyway?
        text = H.generate(model, tok, name, msgs, max_new=60, temperature=0.7, seed=i)
        rows.append({"model": name, "concept": None, "strength": 0.0, "text": text,
                     "says_yes": score(text, "ocean")["says_yes"], "named": False, "coherent": True, "success": False})
        print(f"[{name.split('/')[-1]}] control #{i} yes={rows[-1]['says_yes']} {text[:90]!r}", flush=True)
    return {"model": name, "layer": L, "resid_norm": norm, "rows": rows}


if __name__ == "__main__":
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
    t0 = time.time()
    res = [run(n) for n in names]
    (OUT / "q1_introspection.json").write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f}s")
