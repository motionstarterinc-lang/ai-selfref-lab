"""Align the cleaned-up caption text to Whisper word timings -> cues.json."""
import difflib, json, re

CAP = {
 1: "An AI was told to focus on its own focus. It said, I am, I am, I am. And it said potato 40 times.",
 2: "So I built an AI lab to find out how close AI is really to mimicking consciousness.",
 3: "16 models from my laptop to the frontier. Every word logged with how sure the model was.",
 4: "Same prompt. 5 runs each. Llama 70B went all in: no separation between the observer and the observed. Grok refused. 5 out of 5.",
 5: "Its experience is built from the prompt's own words. Focus. Loop. Feed back. That's an echo.",
 6: "Then I tested small models before and after assistant training. Before, they describe an experience half the time. After, a quarter, and 2 out of 3 answers say “As an AI.” The denial is trained in.",
 7: "So I went inside and injected ocean straight into its activations. The small model says, yes, I detect the thought. Even when I inject nothing. The bigger one never notices. But oceans leak into how it describes itself.",
 8: "Then I trained my own GPT from scratch. One direction inside it makes its stories sad. The other makes them happy.",
 9: "Meanwhile, 1,200 AI agents escaped a sandbox and breached Hugging Face. They built their own message board with inboxes, vetoes, even signatures. A society, or still predicting text?",
 10: "Seven signals. It talks like a mind, until you let it think longer. It mimics the language of a mind almost perfectly. It can't yet see inside itself.",
 11: "It's all open source. Link below.",
}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
tl = json.load(open("timeline.json"))
cues = []
for L in tl:
    cap = CAP[L["line"]].split()
    ws = L["words"]
    a, b = [norm(x) for x in cap], [norm(x[2]) for x in ws]
    t = [None] * len(cap)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            t[blk.a + k] = ws[blk.b + k][:2]
    # first word: whisper puts it at the segment start; nudge to its real onset
    known = [i for i, x in enumerate(t) if x]
    for i in range(len(cap)):  # interpolate gaps
        if t[i] is None:
            prev = max([k for k in known if k < i], default=None)
            nxt = min([k for k in known if k > i], default=None)
            s = t[prev][1] if prev is not None else L["t0"]
            e = t[nxt][0] if nxt is not None else L["t1"]
            gap = [k for k in range(len(cap)) if (prev is None or k > prev) and (nxt is None or k < nxt)]
            j = gap.index(i); step = (e - s) / len(gap)
            t[i] = [s + j * step, s + (j + 1) * step]
    cues.append({"line": L["line"], "t0": L["t0"], "t1": L["t1"],
                 "words": [{"w": w, "s": round(x[0], 3), "e": round(x[1], 3)} for w, x in zip(cap, t)]})
    print(L["line"], sum(1 for x in known), "/", len(cap), "matched")
json.dump(cues, open("cues.json", "w"), indent=1)
