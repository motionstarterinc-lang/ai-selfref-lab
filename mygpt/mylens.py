"""Logit lens + steering for your own GPT (no TransformerLens needed: the model exposes every layer)."""
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import load

HERE = Path(__file__).resolve().parent
CKPT = HERE / "out" / "ckpt.pt"


def available() -> bool:
    return CKPT.exists() and (HERE / "data" / "tokenizer.json").exists()


def load_all():
    tok = Tokenizer.from_file(str(HERE / "data" / "tokenizer.json"))
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    return load(CKPT), tok, {"step": ck["step"], "val_loss": ck["val_loss"]}


@torch.no_grad()
def logit_lens(model, tok, prompt: str, n: int = 24, temperature: float = 0.0) -> dict:
    """Same output shape as interp.lens.logit_lens, so the same heatmap renders it."""
    idx = torch.tensor([tok.encode(prompt).ids])
    full = model.generate(idx, n, temperature=temperature, eos_id=tok.token_to_id("<|end|>"))
    _, _, resid = model(full, return_resid=True)
    start, end = idx.shape[1] - 1, full.shape[1] - 1
    gen = full[0, start + 1:end + 1]
    top, prob, actual = [], [], []
    for r in resid:
        probs = model.layer_logits(r[0, start:end]).softmax(-1)
        p, i = probs.max(-1)
        top.append([tok.decode([t]) for t in i.tolist()])
        prob.append(p.tolist())
        actual.append(probs[torch.arange(len(gen)), gen].tolist())
    return {"tokens": [tok.decode([t]) for t in gen.tolist()], "layers": len(resid),
            "top": top, "prob": prob, "actual": actual, "text": tok.decode(full[0].tolist())}


@torch.no_grad()
def direction(model, tok, pos: list[str], neg: list[str], layer: int) -> torch.Tensor:
    """Mean activation at `layer` on `pos` prompts minus on `neg` prompts."""
    def mean(prompts):
        return torch.stack([model(torch.tensor([tok.encode(p).ids]), return_resid=True)[2][layer][0].mean(0)
                            for p in prompts]).mean(0)
    return mean(pos) - mean(neg)


def steered(model, tok, prompt, vec, layer, coef, n=80, seed=0) -> str:
    torch.manual_seed(seed)  # same seed for steered/unsteered so the only difference is the push
    idx = torch.tensor([tok.encode(prompt).ids])
    out = model.generate(idx, n, eos_id=tok.token_to_id("<|end|>"), steer=(layer, coef * vec) if coef else None)
    return tok.decode(out[0].tolist())
