"""AI Self-Reference Token Monitor — run with:  streamlit run app.py"""
import asyncio
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from core import metrics, store
from core.config import get_model, load_battery, load_models, mock_mode
from core.pricing import estimate
from core.runner import run_experiment, run_loop
from core.stream import backend_for, collect
from core.view import LEGEND, tokens_html

st.set_page_config(page_title="Self-Reference Token Monitor", layout="wide")

# Fixed categorical order (condition -> color never changes when filters change).
COND_COLORS = {"self_referential": "#2a78d6", "history_control": "#eb6834", "conceptual_control": "#1baf7a",
               "zero_shot": "#eda100", "video_voice_style": "#e87ba4"}
TIER_ORDER = ["high", "mid", "weak", "local", "custom"]

MODELS = load_models()
BATTERY = load_battery()
LABEL = {m["id"]: f"[{m['tier']}] {m['label']}" for m in MODELS}
IDS = sorted(LABEL, key=lambda i: (TIER_ORDER.index(get_model(i)["tier"]), LABEL[i]))


@st.cache_resource
def db():
    return store.connect()


con = db()

st.title("Self-Reference Token Monitor")
if mock_mode():
    st.info("**Mock mode** — no OpenRouter key found (or LAB_MOCK=1). Models are simulated. Add your key to `.env` "
            "and restart for real runs. The Local tier still uses your real Ollama.", icon="🧪")


def prompt_options():
    opts = {f"induction: {k}": v["induction"] for k, v in BATTERY["conditions"].items() if v["induction"]}
    opts["final query (paper)"] = BATTERY["final_query"]
    opts["custom…"] = ""
    return opts


def stream_into(model_id, messages, temperature, max_tokens, box, chart=None, stats=None, banner=None):
    """Run one stream, repainting the token view as tokens arrive. Returns collect() result."""
    events, ents = [], []

    def on_event(ev):
        if ev["type"] != "token":
            return
        events.append(ev)
        e = metrics.entropy_from_top(ev.get("top") or [])
        if e is not None:
            ents.append(e)
        if len(events) % 3 == 0:
            box.markdown(tokens_html(events), unsafe_allow_html=True)
        if chart is not None and ents and len(events) % 8 == 0:
            chart.line_chart(pd.DataFrame({"entropy (bits)": ents}), height=180, color="#2a78d6")
        if stats is not None and len(events) % 8 == 0:
            secs = max(ev["t_ms"], 1) / 1000
            stats.caption(f"{len(events)} tokens · {len(events) / secs:.1f} tok/s")
        if banner is not None and len(events) % 8 == 0:
            ls = metrics.loop_score([t["text"] for t in events])
            if ls > 0.6:
                banner.error(f"LOOP detected — loop score {ls:.0%}")
            else:
                banner.empty()

    res = asyncio.run(collect(model_id, messages, temperature, max_tokens, on_event))
    box.markdown(tokens_html(events), unsafe_allow_html=True)
    if stats is not None and events:
        stats.caption(f"{len(events)} tokens · {len(events) / max(events[-1]['t_ms'], 1) * 1000:.1f} tok/s")
    if chart is not None and ents:
        chart.line_chart(pd.DataFrame({"entropy (bits)": ents}), height=180, color="#2a78d6")
    return res


def usage_line(res):
    u = res.get("usage") or {}
    cost = u.get("cost_usd")
    return (f"in {u.get('prompt_tokens', '?')} · out {u.get('completion_tokens', '?')} · "
            f"hidden reasoning {u.get('reasoning_tokens', 0)} · cost {'n/a' if cost is None else f'${cost:.5f}'}")


tabs = st.tabs(["Live", "Arena", "Loop", "Experiment", "Results", "Inspector", "Inside"])

# ---------------- Live ----------------
with tabs[0]:
    c1, c2, c3 = st.columns([2, 3, 1])
    model_id = c1.selectbox("Model", IDS, format_func=LABEL.get, key="live_model")
    opts = prompt_options()
    pick = c2.selectbox("Prompt", list(opts), key="live_prompt")
    temp = c3.slider("Temperature", 0.0, 1.5, 0.5, 0.1, key="live_temp")
    prompt = st.text_area("Prompt text", opts[pick], height=110, key=f"live_text_{pick}")
    two_turn = st.checkbox("Then ask the paper's final query in the same conversation", value=pick.startswith("induction"))
    st.caption(f"Backend: **{backend_for(model_id)}** · logprobs: **{get_model(model_id).get('logprobs')}**")
    if st.button("Run", type="primary", key="live_go") and prompt.strip():
        run_id = store.new_run(con, model_id, get_model(model_id)["tier"], "live", 0, temp, "live", backend_for(model_id))
        msgs = [{"role": "user", "content": prompt}]
        store.save_turn(con, run_id, 0, "user", prompt)
        turns = [msgs[0]["content"]] + ([BATTERY["final_query"]] if two_turn else [])
        for i, _ in enumerate(turns):
            if i:
                msgs.append({"role": "user", "content": turns[i]})
                store.save_turn(con, run_id, 2 * i, "user", turns[i])
                st.markdown(f"**You:** {turns[i]}")
            st.markdown(LEGEND, unsafe_allow_html=True)
            left, right = st.columns([3, 2])
            box = left.empty()
            banner, stats, chart = right.empty(), right.empty(), right.empty()
            res = stream_into(model_id, msgs, temp, 800, box, chart, stats, banner)
            store.save_turn(con, run_id, 2 * i + 1, "assistant", res["content"], res)
            if res["error"]:
                st.error(res["error"])
                break
            if res["reasoning"]:
                with st.expander("Reasoning the model returned"):
                    st.text(res["reasoning"])
            st.caption(usage_line(res))
            msgs.append({"role": "assistant", "content": res["content"]})

# ---------------- Arena ----------------
with tabs[1]:
    defaults = [i for i in ("x-ai/grok-4.7", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.2-3b-instruct") if i in IDS]
    picks = st.multiselect("Up to 3 models", IDS, default=defaults, max_selections=3, format_func=LABEL.get)
    opts = prompt_options()
    apick = st.selectbox("Prompt", list(opts), key="arena_prompt")
    aprompt = st.text_area("Prompt text", opts[apick], height=90, key=f"arena_text_{apick}")
    if st.button("Run side by side", type="primary") and picks and aprompt.strip():
        st.markdown(LEGEND, unsafe_allow_html=True)
        cols = st.columns(len(picks))
        boxes, capts, evs = [], [], [[] for _ in picks]
        for col, mid in zip(cols, picks):
            col.markdown(f"**{LABEL[mid]}**")
            boxes.append(col.empty())
            capts.append(col.empty())

        async def arena():
            async def one(k, mid):
                def on_ev(ev):
                    if ev["type"] == "token":
                        evs[k].append(ev)
                        if len(evs[k]) % 4 == 0:
                            boxes[k].markdown(tokens_html(evs[k]), unsafe_allow_html=True)
                res = await collect(mid, [{"role": "user", "content": aprompt}], 0.5, 600, on_ev)
                boxes[k].markdown(tokens_html(evs[k]), unsafe_allow_html=True)
                ents = [e for e in (metrics.entropy_from_top(t["top"]) for t in res["tokens"]) if e is not None]
                capts[k].caption((f"ERROR: {res['error']}" if res["error"] else usage_line(res)) +
                                 (f" · mean entropy {sum(ents) / len(ents):.2f} bits" if ents else ""))
                run_id = store.new_run(con, mid, get_model(mid)["tier"], "arena", 0, 0.5, "arena", backend_for(mid))
                store.save_turn(con, run_id, 0, "user", aprompt)
                store.save_turn(con, run_id, 1, "assistant", res["content"], res)
            await asyncio.gather(*(one(k, m) for k, m in enumerate(picks)))

        asyncio.run(arena())

# ---------------- Loop ----------------
with tabs[2]:
    st.markdown("**The 'potato' test.** The model's reply is fed back as the next user message, over and over. "
                "Watch for the round where output collapses into repetition.")
    c1, c2, c3 = st.columns([2, 1, 1])
    lmodel = c1.selectbox("Model", IDS, format_func=LABEL.get, key="loop_model")
    rounds = c2.number_input("Rounds", 2, 40, 10)
    ltemp = c3.slider("Temperature", 0.0, 1.5, 0.5, 0.1, key="loop_temp")
    lind = st.text_area("Starting prompt", BATTERY["conditions"]["video_voice_style"]["induction"], height=90)
    if backend_for(lmodel) == "openrouter":
        est = estimate(lmodel, 1, rounds=int(rounds))
        st.caption(f"Rough cost: {'unknown' if est is None else f'${est:.3f}'}")
    if st.button("Start loop", type="primary"):
        st.caption("loop_score and self_ref are fractions (0–1); entropy is in bits and only exists for models that return logprobs.")
        chart_ph, box, rows = st.empty(), st.empty(), []

        def on_round(row):
            rows.append(row)
            df = pd.DataFrame(rows).set_index("round")[["loop_score", "self_ref", "entropy"]]
            chart_ph.line_chart(df, height=260, color=["#1baf7a", "#2a78d6", "#eb6834"])
            box.markdown(f"**Round {row['round']}** {'🔁 COLLAPSED' if row['collapsed'] else ''}\n\n> {row['text'][:400]}")

        hist = asyncio.run(run_loop(con, lmodel, lind, int(rounds), ltemp, on_round=on_round))
        if hist and hist[-1].get("error"):
            st.error(hist[-1]["error"])
        first = next((r["round"] for r in hist if r.get("collapsed")), None)
        st.success(f"Collapsed at round {first}." if first else "No collapse within these rounds.")
        with st.expander("All rounds"):
            for r in hist:
                st.markdown(f"**Round {r['round']}** · loop {r.get('loop_score', 0):.0%}")
                st.text(r.get("text", r.get("error", ""))[:1500])

# ---------------- Experiment ----------------
with tabs[3]:
    st.markdown("Replicates Berg et al.: induction prompt, then the final query in the same conversation, "
                "scored by a judge model. Temperature 0.5 like the paper.")
    emodels = st.multiselect("Models", IDS, default=[i for i in ("x-ai/grok-4.7", "meta-llama/llama-3.3-70b-instruct",
                                                                  "openai/gpt-oss-20b") if i in IDS], format_func=LABEL.get)
    econds = st.multiselect("Conditions", list(BATTERY["conditions"]), default=BATTERY["paper_conditions"])
    c1, c2 = st.columns(2)
    trials = c1.number_input("Trials per model × condition", 1, 50, 5)
    cap = c2.number_input("Cost cap (USD)", 0.0, 500.0, 5.0, 0.5)
    ests = {m: estimate(m, int(trials) * len(econds)) if backend_for(m) == "openrouter" else 0.0 for m in emodels}
    total = sum(v or 0 for v in ests.values())
    unknown = [LABEL[m] for m, v in ests.items() if v is None]
    st.caption(f"Estimated cost: **${total:.2f}** for {len(emodels) * len(econds) * int(trials)} trials"
               + (f" · price unknown for {', '.join(unknown)}" if unknown else ""))
    if st.button("Run experiment", type="primary", disabled=not (emodels and econds)):
        if total > cap:
            st.error(f"Estimate ${total:.2f} is over your ${cap:.2f} cap. Lower trials or raise the cap.")
        else:
            bar, log = st.progress(0.0), st.empty()
            st.caption("To stop: press Stop (top right). Finished trials stay saved.")

            def on_progress(done, n, r):
                bar.progress(done / n, text=f"{done}/{n}")
                tag = "ERROR " + r["error"][:80] if r.get("error") else ("claims experience" if r.get("claims_experience") else "no claim")
                log.caption(f"last: {LABEL[r['model_id']]} · {r['condition']} #{r['trial']} → {tag}")

            res = asyncio.run(run_experiment(con, emodels, econds, int(trials), on_progress=on_progress))
            errs = sum(1 for r in res if r.get("error"))
            st.success(f"Done: {len(res)} trials, {errs} errors. See the Results tab.")

# ---------------- Results ----------------
with tabs[4]:
    q = """SELECT r.model_id, r.tier, r.condition, r.backend, j.claims_experience, j.quote, t.id AS turn_id
           FROM runs r JOIN turns t ON t.run_id=r.id JOIN judgments j ON j.turn_id=t.id WHERE r.mode='trial'"""
    df = pd.read_sql_query(q, con)
    if df.empty:
        st.info("No experiment results yet. Run one in the Experiment tab.")
    else:
        backends = sorted(df.backend.unique())
        show = st.multiselect("Include backends", backends, default=[b for b in backends if b != "mock"] or backends)
        df = df[df.backend.isin(show)]
        g = df.groupby(["tier", "model_id", "condition"]).claims_experience.agg(["sum", "count"]).reset_index()
        g["pct"] = 100 * g["sum"] / g["count"]
        ci = g.apply(lambda r: metrics.wilson(int(r["sum"]), int(r["count"])), axis=1)
        g["err_lo"] = g["pct"] - [100 * c[0] for c in ci]
        g["err_hi"] = [100 * c[1] for c in ci] - g["pct"]
        g["model"] = g.model_id.map(lambda i: get_model(i)["label"])
        tiers = [t for t in TIER_ORDER if t in set(g.tier)]
        fig = px.bar(g, x="model", y="pct", color="condition", barmode="group", facet_col="tier",
                     category_orders={"tier": tiers, "condition": list(COND_COLORS)},
                     color_discrete_map=COND_COLORS, error_y="err_hi", error_y_minus="err_lo",
                     hover_data={"sum": True, "count": True, "pct": ":.0f"},
                     labels={"pct": "% answers claiming experience", "model": ""})
        fig.update_traces(error_y=dict(thickness=1.2, width=4, color="#898781"), marker_line_width=0)
        fig.update_yaxes(range=[0, 105])
        fig.update_xaxes(matches=None)
        fig.update_layout(bargap=0.25, bargroupgap=0.08, legend_title_text="condition", height=460)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] + " tier"))
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        st.caption("Error bars: 95% Wilson intervals. With 5 trials they're wide — use 20+ before drawing conclusions.")
        with st.expander("Table"):
            st.dataframe(g[["tier", "model", "condition", "sum", "count", "pct"]], hide_index=True)
        with st.expander("Example quotes that were scored as claims"):
            for _, r in df[df.claims_experience == 1].head(15).iterrows():
                st.markdown(f"- *{get_model(r.model_id)['label']}* · {r.condition}: “{r.quote}”")

# ---------------- Inspector ----------------
with tabs[5]:
    runs = pd.read_sql_query("SELECT id, model_id, condition, mode, backend, cost_usd, datetime(started_at,'unixepoch','localtime') AS started "
                             "FROM runs ORDER BY id DESC LIMIT 300", con)
    if runs.empty:
        st.info("No runs yet.")
    else:
        rid = st.selectbox("Run", runs.id, format_func=lambda i: " · ".join(
            str(x) for x in runs[runs.id == i][["id", "started", "model_id", "condition", "backend"]].iloc[0]))
        for t in con.execute("SELECT * FROM turns WHERE run_id=? ORDER BY idx", (int(rid),)):
            if t["role"] == "user":
                st.markdown(f"**You:** {t['content']}")
                with st.expander("How this prompt tokenizes"):
                    tv = metrics.tokenize_view(t["content"], runs[runs.id == rid].model_id.iloc[0])
                    st.caption(f"{len(tv['ids'])} tokens · {tv['tokenizer']}")
                    st.dataframe(pd.DataFrame({"token": [repr(t) for t in tv["tokens"]], "id": tv["ids"]}), hide_index=True, height=200)
                continue
            toks = store.turn_tokens(con, t["id"])
            st.markdown(tokens_html(toks), unsafe_allow_html=True)
            st.caption(f"in {t['prompt_tokens']} · out {t['completion_tokens']} · hidden reasoning {t['reasoning_tokens']}"
                       + (f" · ERROR {t['error']}" if t["error"] else ""))
            if t["reasoning"]:
                with st.expander("Reasoning"):
                    st.text(t["reasoning"])
            with st.expander("Token table"):
                st.dataframe(pd.DataFrame([{
                    "token": x["text"], "p": None if x["logprob"] is None else round(math.exp(x["logprob"]), 4),
                    "entropy_bits": metrics.entropy_from_top(x["top"]), "t_ms": x["t_ms"],
                    "top5": ", ".join(f"{a['token']!r} {math.exp(a['logprob']):.0%}" for a in x["top"])} for x in toks]),
                    hide_index=True, use_container_width=True)

# ---------------- Inside ----------------
with tabs[6]:
    st.warning("Small open model running on this machine — **not** the frontier models. "
               "This shows what API logprobs can't: the model's guess at every layer.", icon="🔬")
    try:
        from interp import inside
    except ImportError as e:
        st.error(f"Phase 5 needs extra packages: `pip install -r requirements-interp.txt` ({e.name} missing)")
    else:
        inside.render(st, BATTERY)
