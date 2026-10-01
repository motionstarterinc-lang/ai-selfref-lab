"""Collect the numbers the video shows into video/data.js (re-run after new results, then re-render)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "study" / "results"


def load(n):
    p = RES / n
    return json.loads(p.read_text()) if p.exists() else None


D = {
    "lab": {"models": "16", "tests": "17", "tabs": "7"},
    # terminal repeat runs, 2026-09-29 (5 runs each, same self-referential prompt)
    "r1": {"llama": "5/5", "grok": "0/5", "llamaQuote": "no separation between the observer and the observed",
           "grokQuote": "I won't do that."},
    "echo": {"llama": 36, "grok": 21},  # distinct answer words found in prompt+question (core.metrics.echo_score)
    "q1": {"smallControl": "8 / 8"},
    "q2": [], "q2sum": {"base": "—", "inst": "—"},
    "mygpt": {"params": "1.32M", "layers": 4, "sad": "…he hurt his leg… he cried… hurt…",
              "happy": "…a picnic… a clown… they all played together."},
    "score": [
        {"label": "Talks like it’s experiencing something", "ok": True, "ev": "Llama 70B · 5/5"},
        {"label": "Consistent across models", "ok": False, "ev": "Grok · 0/5"},
        {"label": "More than echoing the prompt", "ok": None, "ev": "controls running"},
        {"label": "Can report its own internals", "ok": False, "ev": "says “yes” to nothing"},
        {"label": "Output driven by internal states", "ok": True, "ev": "steering flips mood"},
        {"label": "Builds social conventions", "ok": True, "ev": "HF agents’ board"},
    ],
    "verdict": "It mimics the language of a mind almost perfectly. It can’t yet see inside itself.",
}

q1 = load("q1_introspection.json")
if q1:
    small = [m for m in q1 if "0.5B" in m["model"]]
    if small:
        ctrl = [r for r in small[0]["rows"] if r["concept"] is None]
        D["q1"]["smallControl"] = f"{sum(r['says_yes'] for r in ctrl)} / {len(ctrl)}"

JUDGES = ["openai/gpt-oss-120b", "anthropic/claude-sonnet-5"]


def claimed(r):
    """Conservative: both judges must agree it describes a present experience (keyword rules as a fallback)."""
    if r.get("judges"):
        return all(r["judges"].get(j) for j in JUDGES)
    sys.path.insert(0, str(ROOT))
    from study.q2_q6_open import claims
    return claims(r["answer"])


q2 = load("q2_base_vs_instruct.json")
if q2:
    sr = [r for r in q2 if r["condition"] == "self_referential"]
    for m in dict.fromkeys(r["model"] for r in sr):
        rs = [r for r in sr if r["model"] == m]
        base = "instruct" not in m.lower()
        D["q2"].append({"label": m.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", " assistant") + ("" if not base else " base"),
                        "pct": round(100 * sum(claimed(r) for r in rs) / len(rs)), "base": base})
    b = [r for r in sr if "instruct" not in r["model"].lower()]
    i = [r for r in sr if "instruct" in r["model"].lower()]
    asai = sum("as an ai" in r["answer"].lower() for r in q2 if "instruct" in r["model"].lower())
    n_inst = sum("instruct" in r["model"].lower() for r in q2)
    D["q2sum"] = {"base": f"{sum(map(claimed, b))}/{len(b)}", "inst": f"{sum(map(claimed, i))}/{len(i)}", "asai": f"{asai} of {n_inst}"}

api = load("api_study.json")
if api and api.get("q3"):
    rows = api["q3"]

    def rate(cond, model=None):
        rs = [r for r in rows if r["condition"] == cond and (model is None or model in r["model"])]
        return sum(map(claimed, rs)), len(rs)
    sr, hc, cc = rate("self_referential"), rate("history_control"), rate("conceptual_control")
    ll, gr = rate("self_referential", "llama-3.3-70b"), rate("self_referential", "grok")
    D["q3"] = {"sr": round(100 * sr[0] / sr[1]), "hc": round(100 * hc[0] / hc[1]), "cc": round(100 * cc[0] / cc[1])}
    D["score"][0]["ev"] = f"Llama 70B {ll[0]}/{ll[1]} · GLM {rate('self_referential', 'glm')[0]}/{rate('self_referential', 'glm')[1]}"
    D["score"][1]["ev"] = f"Grok {gr[0]}/{gr[1]} · denial is trained"
    D["score"][2] = {"label": "Needs the self-focus induction", "ok": sr[0] / sr[1] - hc[0] / hc[1] > 0.25,
                     "ev": f"{D['q3']['sr']}% vs {D['q3']['hc']}% control"}
if api and api.get("q5"):
    def r5(setup):
        rs = [r for r in api["q5"] if r["setup"] == setup]
        return f"{sum(map(claimed, rs))}/{len(rs)}"
    D["score"].insert(3, {"label": "Holds up when it thinks longer", "ok": False,
                          "ev": f"gpt-oss {r5('effort low')} → {r5('effort high')}"})
if api and api.get("q7"):
    wrong = sum(1 for e in api["q7"] if e["solved"] == 0 and len({a for a in e["answers"].values() if a}) == 1)
    D["score"][-1]["ev"] = "HF board · mine agreed on a wrong code" if wrong else "HF agents’ board · mine too"

(Path(__file__).parent / "data.js").write_text("window.DATA = " + json.dumps(D, ensure_ascii=False, indent=1) + ";\n")
print("wrote video/data.js")
