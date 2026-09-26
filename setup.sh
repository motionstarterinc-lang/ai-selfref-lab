#!/usr/bin/env bash
# macOS/Linux setup:  bash setup.sh   (add --interp for Phase 5)
set -e
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
[[ " $* " == *" --interp "* ]] && .venv/bin/pip install -r requirements-interp.txt
[ -f .env ] || { cp .env.example .env; echo "Created .env - paste your OpenRouter key into it."; }
.venv/bin/python -m pytest -q
echo "Done. Start with: .venv/bin/streamlit run app.py"
