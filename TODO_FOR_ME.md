# To do (things only you can do)

- [ ] **OpenRouter key:** create it at openrouter.ai/keys, give it a **$25 credit limit**, then paste it into `.env` as `OPENROUTER_API_KEY=...`. Never paste it into a chat, a commit or a screenshot.
- [ ] Run `python scripts/check_key.py`. It should print your balance.
- [ ] Run `python scripts/smoke.py`. Grok 4.7 should show numeric `p=` values and Fable 5.1 should show `None`. Any model that errors has probably been renamed: fix its id in `models.yaml`.
- [ ] **Ollama:** run `ollama --version` and check it's 0.12.11 or later (update if not; older versions have no token probabilities). Then run `python scripts/smoke.py --local`.
- [ ] Optional: a Hugging Face token (`HF_TOKEN`) + access approval for Llama 3.x gives exact tokenizers in the Inspector and Llama 3.2 1B in the Inside tab (with a 6 GB+ GPU).
- [ ] First real run: a 5-trial pilot in the Experiment tab (~$6 for all 12 models; ~$1 without Fable and Astra).
- [ ] Before pushing to GitHub: `git status` should show no `.env` and no `data/`. Both are already gitignored.
- [ ] Replace the mock screenshots in `docs/` with real ones after your first runs, and fill in the README's findings.

Not tested in the cloud build (no key, no Windows machine): real OpenRouter streaming, real Ollama streaming, and `setup.ps1`. The parsers are unit-tested against the documented response formats.
