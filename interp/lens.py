"""Phase 5: look inside a small open model with TransformerLens.

- logit_lens(): what the model would say if it stopped at each layer, for every generated token.
- self_ref_direction() + steered_generate(): a DIY version of the paper's feature steering. We take the average
  internal state on self-referential prompts minus the average on control prompts, then add (or subtract) that
  direction while the model answers.
"""
import os
from functools import lru_cache

import torch

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"       # ungated, runs on CPU
BIGGER = "meta-llama/Llama-3.2-1B-Instruct"  # gated: needs HF_TOKEN + access approval, wants a GPU


def pick_model() -> str:
    if os.getenv("INTERP_MODEL"):
        return os.environ["INTERP_MODEL"]
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6e9 and os.getenv("HF_TOKEN"):
        return BIGGER
    return SMALL


@lru_cache(maxsize=1)
def load(name: str):
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained_no_processing(name, device=device, dtype=torch.float32)
    model.eval()
    return model


def chat_tokens(model, messages: list[dict]) -> torch.Tensor:
    text = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return model.to_tokens(text, prepend_bos=False)


@torch.no_grad()
def generate(model, messages, max_new_tokens=30, hooks=()):
    toks = chat_tokens(model, messages)
    with model.hooks(fwd_hooks=list(hooks)):
        out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False, verbose=False,
                             stop_at_eos=True, use_past_kv_cache=True)
    return toks, out


@torch.no_grad()
def logit_lens(model, messages, max_new_tokens=24) -> dict:
    """Per generated position and layer: the layer's top guess (top/prob) and the prob it gives the real output (actual)."""
    prompt, full = generate(model, messages, max_new_tokens)
    _, cache = model.run_with_cache(full, names_filter=lambda n: n.endswith("hook_resid_post"))
    start = prompt.shape[1] - 1  # the position that predicts the first generated token
    end = full.shape[1] - 1
    gen_ids = full[0, start + 1:end + 1].tolist()
    top, prob, actual = [], [], []  # actual = prob each layer gives the token finally emitted
    for layer in range(model.cfg.n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, start:end]      # [pos, d_model]
        probs = model.unembed(model.ln_final(resid.unsqueeze(0)))[0].softmax(-1)
        p, idx = probs.max(-1)
        top.append([model.tokenizer.decode([i]) for i in idx.tolist()])
        prob.append(p.tolist())
        actual.append(probs[torch.arange(len(gen_ids)), torch.tensor(gen_ids, device=probs.device)].tolist())
    return {"tokens": [model.tokenizer.decode([i]) for i in gen_ids], "layers": model.cfg.n_layers,
            "top": top, "prob": prob, "actual": actual, "text": model.tokenizer.decode(gen_ids)}


@torch.no_grad()
def self_ref_direction(model, self_prompts: list[str], control_prompts: list[str], layer: int) -> torch.Tensor:
    """Mean residual activation (over prompt positions) on self-referential prompts minus on controls."""
    def mean_act(prompts):
        acts = []
        for p in prompts:
            _, cache = model.run_with_cache(chat_tokens(model, [{"role": "user", "content": p}]),
                                            names_filter=f"blocks.{layer}.hook_resid_post")
            acts.append(cache[f"blocks.{layer}.hook_resid_post"][0].mean(0))
        return torch.stack(acts).mean(0)
    return mean_act(self_prompts) - mean_act(control_prompts)


def steered_generate(model, messages, direction: torch.Tensor, layer: int, coef: float, max_new_tokens=60) -> str:
    def add(resid, hook):
        return resid + coef * direction
    hooks = [(f"blocks.{layer}.hook_resid_post", add)] if coef else []
    prompt, out = generate(model, messages, max_new_tokens, hooks)
    return model.tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)
