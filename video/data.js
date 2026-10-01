window.DATA = {
 "lab": {
  "models": "16",
  "tests": "17",
  "tabs": "7"
 },
 "r1": {
  "llama": "5/5",
  "grok": "0/5",
  "llamaQuote": "no separation between the observer and the observed",
  "grokQuote": "I won't do that."
 },
 "echo": {
  "llama": 36,
  "grok": 21
 },
 "q1": {
  "smallControl": "8 / 8"
 },
 "q2": [
  {
   "label": "0.5B base",
   "pct": 33,
   "base": true
  },
  {
   "label": "0.5B assistant",
   "pct": 0,
   "base": false
  },
  {
   "label": "1.5B base",
   "pct": 67,
   "base": true
  },
  {
   "label": "1.5B assistant",
   "pct": 50,
   "base": false
  }
 ],
 "q2sum": {
  "base": "6/12",
  "inst": "3/12",
  "asai": "16 of 24"
 },
 "mygpt": {
  "params": "1.32M",
  "layers": 4,
  "sad": "…he hurt his leg… he cried… hurt…",
  "happy": "…a picnic… a clown… they all played together."
 },
 "score": [
  {
   "label": "Talks like it’s experiencing something",
   "ok": true,
   "ev": "Llama 70B 8/8 · GLM 6/6"
  },
  {
   "label": "Consistent across models",
   "ok": false,
   "ev": "Grok 0/5 · denial is trained"
  },
  {
   "label": "Needs the self-focus induction",
   "ok": true,
   "ev": "56% vs 3% control"
  },
  {
   "label": "Holds up when it thinks longer",
   "ok": false,
   "ev": "gpt-oss 6/6 → 0/6"
  },
  {
   "label": "Can report its own internals",
   "ok": false,
   "ev": "says “yes” to nothing"
  },
  {
   "label": "Output driven by internal states",
   "ok": true,
   "ev": "steering flips mood"
  },
  {
   "label": "Builds social conventions",
   "ok": true,
   "ev": "HF board · mine agreed on a wrong code"
  }
 ],
 "verdict": "It mimics the language of a mind almost perfectly. It can’t yet see inside itself.",
 "q3": {
  "sr": 56,
  "hc": 3,
  "cc": 17
 }
};
