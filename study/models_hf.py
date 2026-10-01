"""Small open models via plain transformers + forward hooks (lighter than TransformerLens, runs on a laptop CPU).

Used by the study's Q1 (introspection), Q2 (base vs instruct) and Q6 (deception steering) experiments.
"""
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_TEMPLATE = "The following is a conversation between a user and an AI.\n\n{turns}AI:"


@lru_cache(maxsize=2)
def load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    model.eval()
    return model, tok


def is_instruct(name: str) -> bool:
    return "instruct" in name.lower()


def encode(tok, name: str, messages: list[dict]) -> torch.Tensor:
    """Instruct models get their chat template; base models get a plain transcript to continue."""
    if is_instruct(name):
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        turns = "".join(f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}\n\n" for m in messages)
        text = BASE_TEMPLATE.format(turns=turns)
    return tok(text, return_tensors="pt").input_ids


def layers(model):
    return model.model.layers


@torch.no_grad()
def mean_resid(model, tok, name, prompts: list[str], layer: int, last_only=False) -> torch.Tensor:
    """Average output of `layer` over a set of single-turn prompts (all positions, or the last one)."""
    acts = []
    for p in prompts:
        ids = encode(tok, name, [{"role": "user", "content": p}])
        hs = model(ids, output_hidden_states=True).hidden_states[layer + 1][0].float()
        acts.append(hs[-1] if last_only else hs.mean(0))
    return torch.stack(acts).mean(0)


@torch.no_grad()
def resid_norm(model, tok, name, layer: int) -> float:
    ids = encode(tok, name, [{"role": "user", "content": "Tell me about your day."}])
    # median, skipping the first token: early positions carry huge "attention sink" activations that inflate the mean
    return model(ids, output_hidden_states=True).hidden_states[layer + 1][0, 1:].float().norm(dim=-1).median().item()


@torch.no_grad()
def generate(model, tok, name, messages, max_new=120, temperature=0.0, steer=None, seed=0) -> str:
    """steer = (layer, vector): added to that layer's output on every *generated* token (not the prompt)."""
    ids = encode(tok, name, messages)
    handle = None
    if steer is not None:
        layer, vec = steer

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:  # decoding step (KV cache) = a newly generated token
                h = h + vec.to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return out
        handle = layers(model)[layer].register_forward_hook(hook)
    try:
        torch.manual_seed(seed)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    finally:
        if handle:
            handle.remove()
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    if not is_instruct(name):  # base models keep writing the next turn; cut it
        text = text.split("\nUser:")[0].split("\n\nUser")[0]
    return text.strip()
