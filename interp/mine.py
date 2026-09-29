"""'My GPT' section of the Inside tab: the model you trained yourself in Phase 6."""
import html
import sys
from pathlib import Path

import pandas as pd

MYGPT = Path(__file__).resolve().parent.parent / "mygpt"
sys.path.insert(0, str(MYGPT))
import mylens  # noqa: E402  (mygpt/mylens.py)

HAPPY = ["Lily was so happy. She smiled and laughed and hugged her mom.",
         "The sun was bright and everyone was having fun at the park.",
         "Tom got a new puppy and he was very excited and glad."]
SAD = ["Lily was very sad. She cried because she lost her toy.",
       "It was a dark rainy day and Tom felt lonely and scared.",
       "The little bird was hurt and nobody came to help it."]


def render(st, heatmap):
    if not mylens.available():
        st.info("No trained model yet. In a terminal: `python mygpt/prepare.py` then `python mygpt/train.py` "
                "(~1 h on a laptop CPU). Then come back here.")
        return
    model, tok, meta = st.cache_resource(show_spinner="Loading your GPT…")(mylens.load_all)()
    c = model.cfg
    st.caption(f"**{model.n_params() / 1e6:.2f}M parameters** · {c.n_layer} layers · {c.n_head} heads · "
               f"{c.n_embd}-dim · vocab {c.vocab_size} · trained {meta['step']} steps · val loss {meta['val_loss']:.2f}")

    log = MYGPT / "out" / "log.csv"
    if log.exists():
        df = pd.read_csv(log).dropna(subset=["val_loss"])
        with st.expander("Training curve (validation loss — lower is better)"):
            st.line_chart(df.set_index("step")[["val_loss"]], height=200, color="#2a78d6")
            st.caption("Starts near ln(vocab) ≈ 8.3, which is pure guessing. Each drop is the model learning "
                       "something: spelling, then grammar, then story shape.")

    st.subheader("1 · Logit lens on your model")
    prompt = st.text_input("Story start", "Once upon a time, there was a little")
    n = st.slider("Tokens to generate", 8, 48, 24, key="my_n")
    if st.button("Run logit lens on my GPT"):
        st.session_state["my_lens"] = mylens.logit_lens(model, tok, prompt, n)
    if "my_lens" in st.session_state:
        r = st.session_state["my_lens"]
        st.plotly_chart(heatmap(r, "my GPT"), use_container_width=True)
        st.caption(f"Output: {html.escape(r['text'])}")

    st.subheader("2 · Steering your model: happy ↔ sad")
    st.markdown("Same trick as above, but on a model you built: average its internal state on happy sentences, "
                "subtract sad ones, and push that direction while it writes. Every number in that vector is yours.")
    layer = st.slider("Layer", 0, c.n_layer - 1, c.n_layer // 2, key="my_layer")
    coef = st.slider("Strength (negative = sadder)", -6.0, 6.0, 0.0, 0.5, key="my_coef")
    sp = st.text_input("Prompt", "One day, Ben went to the", key="my_sp")
    if st.button("Write with this steering"):
        vec = mylens.direction(model, tok, HAPPY, SAD, layer)
        c1, c2 = st.columns(2)
        c1.markdown("**No steering**")
        c1.write(mylens.steered(model, tok, sp, vec, layer, 0.0))
        c2.markdown(f"**Steered × {coef:+}** at layer {layer}")
        c2.write(mylens.steered(model, tok, sp, vec, layer, coef))
