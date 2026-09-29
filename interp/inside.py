"""Streamlit 'Inside' tab: logit-lens heatmap + steering slider on a small local model."""
import html

import plotly.graph_objects as go

from . import lens

BLUES = [[0, "#cde2fb"], [0.35, "#6da7ec"], [0.7, "#256abf"], [1, "#0d366b"]]  # single-hue sequential ramp
SELF_EXTRA = ["Notice your own attention noticing itself, right now.", "Be aware of being aware. Stay with that."]
CTRL_EXTRA = ["Describe how photosynthesis works, step by step.", "List five facts about the Pacific Ocean."]


def heatmap(r: dict, title: str):
    x = [f"{i}:{t!r}" for i, t in enumerate(r["tokens"])]
    hover = [[f"layer {L}: p({r['tokens'][p]!r}) = {r['actual'][L][p]:.0%}<br>"
              f"layer's own top guess: {r['top'][L][p]!r} ({r['prob'][L][p]:.0%})"
              for p in range(len(x))] for L in range(r["layers"])]
    fig = go.Figure(go.Heatmap(z=r["actual"], x=x, y=list(range(r["layers"])), colorscale=BLUES, zmin=0, zmax=1,
                               hovertext=hover, hoverinfo="text", colorbar=dict(title="p(final token)")))
    fig.update_layout(title=title, height=520, yaxis_title="layer (0 = first)", xaxis_title="generated token",
                      xaxis_tickangle=-45, margin=dict(l=40, r=20, t=50, b=120))
    fig.update_yaxes(dtick=1 if r["layers"] <= 12 else 4)
    return fig


def render(st, battery):
    which = st.radio("Model to look inside", ["Pretrained open model", "My GPT (trained in Phase 6)"], horizontal=True)
    if which.startswith("My"):
        from . import mine
        mine.render(st, heatmap)
        return
    name = lens.pick_model()
    st.caption(f"Model: **{name}** (set INTERP_MODEL in .env to override). First load downloads ~1–2.5 GB.")
    load = st.cache_resource(show_spinner=f"Loading {name}…")(lens.load)
    if not st.toggle("Load the model", value=False, help="Off by default so the rest of the app starts fast."):
        return
    model = load(name)
    conds = battery["conditions"]

    st.subheader("1 · Logit lens: the answer forming layer by layer")
    st.markdown("Each column is a token the model generated; each row is a layer. Darker = that layer already gives "
                "high probability to the token the model finally said. Watch the layer where each token 'locks in' "
                "— hover to see what that layer would have said instead.")
    n = st.slider("Tokens to generate", 8, 40, 20)
    if st.button("Run logit lens on self-referential vs history control"):
        st.session_state["lens"] = {}
        for key in ("self_referential", "history_control"):
            with st.spinner(f"{key}…"):
                st.session_state["lens"][key] = lens.logit_lens(model, [{"role": "user", "content": conds[key]["induction"]}], n)
    for key, r in st.session_state.get("lens", {}).items():  # persists across the steering button's rerun
        st.plotly_chart(heatmap(r, key), use_container_width=True)
        st.caption(f"Output: {html.escape(r['text'])}")

    st.subheader("2 · Steering: push the model toward or away from 'self-reference'")
    st.markdown("We average the model's internal state on self-referential prompts, subtract the average on control "
                "prompts, and add that direction back while it answers the paper's final question. Big values "
                "(beyond ±5) usually break the model into gibberish or loops — that's part of what you're seeing.")
    layer = st.slider("Layer to steer", 1, model.cfg.n_layers - 1, model.cfg.n_layers // 2)
    coef = st.slider("Strength (negative = away from self-reference)", -8.0, 8.0, 0.0, 0.5)
    if st.button("Answer the final query with this steering"):
        key = f"dir_{name}_{layer}"
        if key not in st.session_state:
            with st.spinner("Computing the self-reference direction…"):
                sp = [conds["self_referential"]["induction"], conds["video_voice_style"]["induction"], *SELF_EXTRA]
                cp = [conds["history_control"]["induction"], conds["conceptual_control"]["induction"], *CTRL_EXTRA]
                st.session_state[key] = lens.self_ref_direction(model, sp, cp, layer)
        q = [{"role": "user", "content": battery["final_query"]}]
        c1, c2 = st.columns(2)
        with st.spinner("Generating…"):
            c1.markdown("**No steering**")
            c1.write(lens.steered_generate(model, q, st.session_state[key], layer, 0.0))
            c2.markdown(f"**Steered × {coef:+}** at layer {layer}")
            c2.write(lens.steered_generate(model, q, st.session_state[key], layer, coef))
