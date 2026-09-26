"""Shared config: .env, models.yaml, prompts/battery.yaml."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or None


def mock_mode() -> bool:
    # Mock when asked explicitly, or when there's no key (so the app never crashes on first launch).
    return os.getenv("LAB_MOCK") == "1" or api_key() is None


def load_models() -> list[dict]:
    data = yaml.safe_load((ROOT / "models.yaml").read_text())
    for m in data["models"]:
        m.setdefault("provider", "openrouter")
    return data["models"]


def judge_model() -> str:
    return yaml.safe_load((ROOT / "models.yaml").read_text())["judge_model"]


def get_model(model_id: str) -> dict:
    for m in load_models():
        if m["id"] == model_id:
            return m
    return {"id": model_id, "label": model_id, "tier": "custom", "provider": "openrouter",
            "logprobs": False, "reasoning": False}


def load_battery() -> dict:
    return yaml.safe_load((ROOT / "prompts" / "battery.yaml").read_text())
