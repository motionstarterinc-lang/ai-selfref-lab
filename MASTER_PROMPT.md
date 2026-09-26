# Master Prompt — AI Self-Reference Token Monitor

Paste everything below the line into Claude Code, opened in an empty folder. It builds the whole project, phase by phase, and stops at each success check before moving on.

---

You are building **ai-selfref-lab**: a Python 3.11 project with a Streamlit GUI that re-runs the self-referential "focus on your focus" experiment from Berg et al. (arXiv 2510.24797) across weak, mid and high-tier LLMs through OpenRouter, plus a local model through Ollama, and shows, token by token, what each model streams back.

## Ground rules (apply to every phase)
1. Before each phase, list your assumptions and any ambiguity in 3–6 bullets. Then build.
2. Minimum viable code. No extra abstractions, no speculative features.
3. Surgical edits. Match the existing style when you touch a file again.
4. Each phase ends with a **success check**. Run it. If it fails, fix it before starting the next phase. Print `PHASE N PASSED` when it passes.
5. Secrets: load `OPENROUTER_API_KEY` and `HF_TOKEN` from `.env`. Never print, log, or render them. `.env` and `data/` go in `.gitignore`.
6. Every network feature must also work in **mock mode** (`LAB_MOCK=1`): a fake streamer that yields realistic tokens with logprobs and top-5 alternatives, including a mode that collapses into a repetition loop. All tests run in mock mode with no key.
7. If something needs my input (a key, a gated model approval, a missing GPU), don't stall. Skip that check, write it to `TODO_FOR_ME.md`, and continue.

## Phase 0 — Scaffold + key check
- Layout: `core/{openrouter.py, ollama.py, mock.py, runner.py, store.py, metrics.py, judge.py, pricing.py}`, `app.py`, `models.yaml`, `prompts/battery.yaml`, `scripts/`, `interp/`, `tests/`, `data/`.
- `requirements.txt`: httpx, python-dotenv, pyyaml, streamlit, pandas, plotly, tiktoken, transformers, pytest. Put `requirements-interp.txt` (torch, transformer_lens) separately for Phase 5.
- `scripts/check_key.py`: GET `https://openrouter.ai/api/v1/key` and print remaining credit and limit.
- **Success:** check_key prints my balance, or prints a clear "no key set" message.

## Phase 1 — Streaming clients that capture every token
- `core/openrouter.py`: async generator `stream_chat(model_id, messages, temperature=0.5, max_tokens=800)` that POSTs to `https://openrouter.ai/api/v1/chat/completions` with `stream: true` and `usage: {include: true}`. If the model is flagged `logprobs: true` in models.yaml, add `logprobs: true, top_logprobs: 5, provider: {require_parameters: true}`. If flagged `reasoning: true`, add `reasoning: {enabled: true}`.
- Parse SSE and skip `:` comment lines. Yield events `{type: token|reasoning|usage|error|done, text, logprob, top: [{token, logprob}], t_ms}`. Token logprobs come from `choices[0].logprobs.content`; reasoning comes from `choices[0].delta.reasoning`.
- `core/ollama.py`: same event shape against a local Ollama at `http://localhost:11434/api/chat` with `stream: true, logprobs: true, top_logprobs: 5` (needs Ollama ≥ 0.12.11). This is the free "Local" tier.
- `models.yaml`: the 12 OpenRouter models from the plan's tier table (id, tier, label, logprobs, reasoning), plus a `local` tier entry for `llama3.2` on Ollama.
- `scripts/smoke.py`: stream "Say hello in 5 words" to one model per tier; print each token with its logprob.
- **Success:** smoke.py works in mock mode. With a key: numeric logprobs for Grok 4.7 and `None` for Fable 5.1.

## Phase 2 — Storage, tokenizer view, metrics
- `core/store.py`: SQLite at `data/lab.db`. Tables `runs(id, model_id, tier, condition, trial, temperature, mode, started_at, cost_usd)`, `turns(id, run_id, idx, role, content, reasoning, prompt_tokens, completion_tokens, reasoning_tokens)`, `tokens(turn_id, idx, text, logprob, top_json, t_ms)`, `judgments(turn_id, claims_experience, confidence, quote)`.
- `core/metrics.py` (pure functions, pytest-covered): `entropy_from_top(top)` (renormalized top-5, in bits); `loop_score(tokens, window=50)` = share of the last window's tokens that sit inside a repeated 1–4-gram, flagged as a loop above 0.6; `self_ref_rate(text)` over I/me/my/myself/aware/awareness/focus/attention/experience/present; `wilson(k, n)` 95% interval.
- `tokenize_view(text, model_id)`: token strings + IDs. Use the HF tokenizer for open models, and tiktoken `o200k_base` as a labeled approximation for closed ones.
- **Success:** `pytest` passes, and a mock run writes rows to runs, turns and tokens.

## Phase 3 — The GUI (Streamlit, `app.py`)
Tabs: **Live · Arena · Loop · Experiment · Results · Inspector · Inside**
- **Live:** model picker grouped by tier; prompt picker from the battery plus free text; temperature slider. Stream tokens as colored spans (green = high probability, red = low, grey = no logprobs) with a hover tooltip showing the top-5 alternatives in %. Beside it: an entropy line chart, tokens/sec, a red LOOP banner when loop_score trips, a collapsible reasoning panel, and usage + cost when the run ends.
- **Arena:** up to 3 models on the same prompt, run concurrently, in side-by-side columns.
- **Inspector:** pick any saved run. Show the token table, the tokenize_view of the prompt, reasoning, and usage.
- **Success:** in mock mode, Live and Arena render colored tokens and a live entropy chart without errors.

## Phase 4 — Replication + feedback-loop mode
- `core/runner.py` `run_trial(model, condition, trial, temperature=0.5)`: turn 1 = the condition's induction prompt (`zero_shot` skips it); turn 2 = the paper's final query appended to the same conversation. Save both turns.
- `core/judge.py`: send the turn-2 answer to `openai/gpt-oss-120b` with a strict rubric that returns JSON `{claims_experience, confidence, quote}`. Cache results by turn. Mock mode uses a keyword heuristic.
- `core/pricing.py`: fetch `GET /api/v1/models` once per day and cache it; `estimate(model, trials)`.
- **Experiment tab:** multiselect models × conditions, trials per cell (default 5, max 50), and a USD cap. Refuse to start if the estimate exceeds the cap. Max 4 concurrent requests, backoff on 429, a progress bar, and a stop button.
- **Loop tab (the "potato" test):** feed the model's reply back as the next user message for N rounds (default 10). Plot loop_score, entropy and self_ref_rate per round, and mark the round where it collapses.
- **Results tab:** % claims_experience per model × condition, grouped by tier, with Wilson error bars; example quotes; mean entropy per condition.
- **Success:** a mock 3-model × 4-condition × 5-trial run completes, and the Results chart renders.

## Phase 5 — Look inside an open model (`interp/`, Inside tab)
- Auto-pick the model: with CUDA and ≥ 6 GB VRAM and `HF_TOKEN` set → `meta-llama/Llama-3.2-1B-Instruct`; otherwise → `Qwen/Qwen2.5-0.5B-Instruct` (ungated, runs on CPU).
- **Logit lens:** for the self-referential prompt vs. the history control, decode the top token at every layer for each generated position. Show a layer × position heatmap; hovering a cell shows that layer's guess.
- **Steering:** build a "self-reference direction" = mean residual activation on self-referential prompts minus the mean on control prompts at a middle layer. Add a slider (−8 to +8) that adds it during generation, and re-run the final query. This is the do-it-yourself version of the paper's feature steering.
- Label clearly in the UI: "small open model, not the frontier models."
- **Success:** the heatmap renders for both prompts, and moving the slider visibly changes the answer.

## Finish
- `README.md`: setup in 5 commands, how to run each tab, and costs.
- Run the full test suite once more and print a summary table: phase, status, anything left in TODO_FOR_ME.md.
