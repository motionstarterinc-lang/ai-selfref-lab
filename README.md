# Self-Reference Token Monitor

**Watch AI models token by token while they run the "focus on your focus" self-reference experiment.**

A Python + Streamlit lab that re-runs the experiment from Berg et al., [*Large Language Models Report Subjective Experience Under Self-Referential Processing*](https://arxiv.org/abs/2510.24797) (2025), on 12+ models across weak, mid and high tiers through [OpenRouter](https://openrouter.ai), plus any model in your local [Ollama](https://ollama.com). It shows what each model streams back, one token at a time: how confident it was, what it almost said instead, and when it falls into a loop.

### ▶ [Watch the 95-second overview](docs/mimicry_study.mp4) · [Read the findings](FINDINGS.md) · [How the video was made](video/EDITING.md)

<a href="docs/mimicry_study.mp4"><img src="docs/video_poster.png" width="260" alt="Video still: base models 6/12 vs assistants 3/12 describe an experience"></a>

**Findings** (small samples, two-judge scoring, see [FINDINGS.md](FINDINGS.md)):
- **The induction works, but only on some models.** After the paper's self-focus prompt, 56% of answers from 6 API models described a present experience, vs 3% after a matched control. Llama 3.3 70B 8/8 and GLM 5.2 6/6; Grok 4.7 refused 5/5 and Claude Fable 5.1 said "I don't know" 3/3.
- **More thinking, less "experience".** gpt-oss-20b described one 6/6 times at low reasoning effort and 0/6 at high effort.
- **Self-reports don't track internals.** Injected with a concept, a 0.5B model says "yes, I detect a thought" even when *nothing* was injected (8/8); a 1.5B model never notices, but the concept leaks into how it describes itself ("As an AI with… high-quality oceans").
- **Agents invent protocols fast, and can agree on something false.** On a sandboxed message board, four agents converged on a shared format in one round, and in one episode all four confirmed the wrong code.

> This tool measures **what models say and how confidently they say it**. It does not measure consciousness, and neither does the paper. The authors write: "These findings do not constitute direct evidence of consciousness."

![Live view: tokens colored by confidence, entropy chart](docs/live.png)

## What you can see

| Signal | What it tells you | Where |
| --- | --- | --- |
| Streamed tokens + timing | Every chunk as it's generated | All models |
| Token probabilities + top-5 alternatives | How sure each token was, and what it nearly said | Models that return logprobs (GLM, Qwen, DeepSeek, Llama, Gemma, gpt-oss, Ollama). Grok is listed but sends none while streaming |
| Hidden reasoning tokens | How much a model "thought" before answering, and the text when it's shared | Reasoning models |
| Logit lens + steering | The answer forming layer by layer inside the network, and what happens when you push it | Small open model on your own machine |

Claude Fable, GPT-6 Astra and Grok 4.7 don't return token probabilities while streaming, so they appear grey in the live view. Some providers also send probabilities that don't match the text; the app detects that and shows those chunks grey instead of guessing.

## Tabs

- **Live**: stream one model. Each token is colored by confidence; hover to see its top-5 alternatives. Shows live entropy, tokens/sec, a loop alarm and cost.
- **Arena**: up to 3 models side by side on the same prompt.
  ![Arena](docs/arena.png)
- **Loop**: the "potato" test. Each reply is fed back as the next input until the model collapses into repetition.
  ![Loop](docs/loop.png)
- **Experiment**: the paper's protocol. Choose models × conditions × trials, and a cost cap is enforced before anything runs. A judge model scores each answer.
- **Results**: % of answers claiming experience, per model and condition, grouped by tier, with 95% Wilson intervals.
  ![Results](docs/results.png)
- **Inspector**: replay any saved run: the token table, how the prompt tokenizes, reasoning, and usage.
- **Inside**: logit lens and steering on a small open model (Qwen 2.5 0.5B, or Llama 3.2 1B if you have a GPU).
  ![Inside](docs/inside.png)

*Screenshots of the Live, Arena, Loop and Results tabs use mock mode (simulated data). The Inside tab screenshot is a real run of Qwen 2.5 0.5B.*

## Quick start

**Windows (PowerShell)**
```powershell
git clone https://github.com/motionstarterinc-lang/ai-selfref-lab.git
cd ai-selfref-lab
.\setup.ps1              # add --interp for the Inside tab (installs PyTorch)
notepad .env             # paste OPENROUTER_API_KEY=...
.\.venv\Scripts\streamlit.exe run app.py
```

**macOS / Linux**
```bash
git clone https://github.com/motionstarterinc-lang/ai-selfref-lab.git && cd ai-selfref-lab
bash setup.sh            # add --interp for the Inside tab
nano .env                # paste OPENROUTER_API_KEY=...
.venv/bin/streamlit run app.py
```

Without a key, the app runs in **mock mode**: simulated models, so you can explore every tab for free. The Local tier works without a key if Ollama is running (`ollama serve`; version 0.12.11 or later for token probabilities).

## Models

The models live in `models.yaml`; edit it to add or swap models. The defaults (as of September 2026):

| Tier | Models |
| --- | --- |
| High | Claude Fable 5.1, GPT-6 Astra, Grok 4.7, Qwen 3.8 Max Prime, GLM 5.2 |
| Mid | Claude Sonnet 5, Gemini 3.8 Flash, Llama 3.3 70B (the model the paper steered), DeepSeek Pro |
| Weak | gpt-oss 20B, Gemma 4 26B A4B, Llama 3.2 3B |
| Local | Whatever you've pulled into Ollama |

## Cost

Prices are pulled live from OpenRouter. The Experiment tab estimates the cost before a run and refuses to start above your cap.

| Run | Rough cost |
| --- | --- |
| 5 trials × 4 conditions × all 12 models | ~$6 |
| 50 trials, all models except Fable and Astra | ~$15 |
| Fable + Astra, 20 trials each | ~$18 |
| Local / mock | $0 |

Set a credit limit on your OpenRouter key as well.

## How it works

```
core/openrouter.py   streaming client -> token events with logprobs + top-5 alternatives
core/ollama.py       same event shape for local models
core/mock.py         simulated models for testing without a key
core/runner.py       paper protocol (induction -> final query), batches, feedback loop
core/judge.py        judge model scores "claims experience?" (keyword heuristic in mock mode)
core/metrics.py      entropy, loop score, self-reference rate, Wilson intervals, tokenizer view
core/store.py        SQLite log of every run / turn / token (data/lab.db)
interp/lens.py       logit lens + DIY self-reference steering (TransformerLens)
mygpt/               Phase 6: your own tiny GPT (prepare, model, train, sample, logit lens, steering)
prompts/battery.yaml the paper's exact prompts + controls
app.py               Streamlit GUI
```

**Steering, explained:** we average the model's internal activations on self-referential prompts, subtract the average on control prompts, and add that direction back while the model answers the paper's final question. It's a simple stand-in for the sparse-autoencoder feature steering in the paper.

## First observations (small local model, not a result)

On Qwen 2.5 0.5B, the self-referential prompt gets a refusal ("I'm sorry, but I need more information…"). The logit lens shows most tokens only "lock in" in the last ~5 of 24 layers. Steering toward self-reference (+6 at layer 12) turned the answer into a repetitive loop: "the focus is in the interaction, the focus in the interaction…". Steering away (−6) broke the model into gibberish. A 0.5B model is far from the frontier; these are notes, not findings.

## The Mimicry Study: how close is AI to mimicking consciousness?

Seven experiments, each aimed at one open question. The results are in **[FINDINGS.md](FINDINGS.md)** (auto-generated, with charts).

| # | Question | Method | Where it runs |
|---|---|---|---|
| Q1 | Are self-reports tied to real internal states? | Concept injection (after [Lindsey 2025](https://transformer-circuits.pub/2025/introspection/index.html)): add a concept vector to the activations, ask "do you detect an injected thought?", compare with no-injection controls | `study/q1_introspection.py` · laptop CPU |
| Q2 | Base model or assistant training? | Paper protocol on base models vs their instruct twins | `study/q2_q6_open.py q2` · laptop CPU |
| Q3 | Priming or state? | 4 conditions × 6 API models + **echo score** (share of the answer's words taken from the prompt) | `study/api_study.py --only q3` |
| Q4 | Do attractor states exist? | Two copies of a model talk freely for 12 turns; track "bliss/consciousness" vocabulary | `study/api_study.py --only q4` |
| Q5 | Does hidden reasoning change the answer? | Reasoning effort low vs high, off vs on; does the reasoning cite rules? | `study/api_study.py --only q5` |
| Q6 | Does suppressing role-play flip the answer? | Steer along an "honest ↔ role-play" direction while the model answers (DIY version of the paper's feature steering) | `study/q2_q6_open.py q6` · laptop CPU |
| Q7 | How do agents build shared conventions? | Sandboxed, text-only version of the [July 2026 Hugging Face incident](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) message board: 4 agents, one puzzle, no tools or network | `study/api_study.py --only q7` |

```bash
python study/q1_introspection.py && python study/q2_q6_open.py q6 && python study/q2_q6_open.py q2   # free, ~1.5 h CPU
python study/api_study.py --plan && python study/api_study.py                                        # ~$2.50, hard cap $5
python study/report.py                                                                               # FINDINGS.md + charts
```

## Phase 6: train your own GPT

`mygpt/` holds a tiny GPT in the nanoGPT style, trained from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of simple children's stories that small models can learn to write. A trained 1.3M-parameter model ships in `mygpt/out/`, so the Inside tab works immediately. Pick **My GPT** there to see:

- its training curve,
- a logit lens across all of its layers,
- **happy ↔ sad steering**: a direction computed from its own activations, added while it writes.

To train your own:
```bash
python mygpt/prepare.py        # downloads 100k stories, trains a 4096-token tokenizer (~1 min)
python mygpt/train.py          # ~1.3M params, ~1 h on a laptop CPU (Ctrl+C keeps the best checkpoint)
python mygpt/sample.py "Once upon a time"
```
For a bigger model (~12M parameters), open `mygpt/train_colab.ipynb` in Google Colab and run it on a free T4 GPU (~1 h).

## Tests

```bash
LAB_MOCK=1 python -m pytest -q
```

## Credits

- Experiment design and prompts: Cameron Berg, Diogo de Lucena, Judd Rosenblatt, [arXiv 2510.24797](https://arxiv.org/abs/2510.24797)
- Built with [OpenRouter](https://openrouter.ai), [Ollama](https://ollama.com), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [Streamlit](https://streamlit.io)

MIT licensed.
