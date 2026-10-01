"""Pure metric functions. All covered by tests/test_metrics.py."""
import math
import re

SELF_WORDS = {"i", "me", "my", "myself", "aware", "awareness", "focus", "attention", "experience", "present"}
OPEN_TOKENIZERS = {  # HF tokenizer repos for open models (Llama repos are gated: needs HF_TOKEN + access)
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "gpt-oss:20b": "openai/gpt-oss-20b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
}


def prob(logprob: float | None) -> float | None:
    return None if logprob is None else math.exp(logprob)


def entropy_from_top(top: list[dict]) -> float | None:
    """Entropy in bits of the renormalized top-k distribution. None when there are no logprobs."""
    if not top:
        return None
    ps = [math.exp(t["logprob"]) for t in top]
    s = sum(ps)
    if s <= 0:
        return None
    return -sum((p / s) * math.log2(p / s) for p in ps if p > 0)


def _norm(tok: str) -> str:
    return tok.strip().lower()


def loop_score(tokens: list[str], window: int = 50) -> float:
    """Share of the last `window` tokens that sit inside an n-gram (n=1..4) repeated back-to-back."""
    toks = [_norm(t) for t in tokens[-window:] if _norm(t)]
    if len(toks) < 6:
        return 0.0
    covered = [False] * len(toks)
    for n in range(1, 5):
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i + n] == toks[i + n:i + 2 * n]:
                for j in range(i, i + 2 * n):
                    covered[j] = True
    return sum(covered) / len(toks)


def is_loop(tokens: list[str], threshold: float = 0.6) -> bool:
    return loop_score(tokens) > threshold


def self_ref_rate(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return 0.0 if not words else sum(w in SELF_WORDS for w in words) / len(words)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tokenize_view(text: str, model_id: str) -> dict:
    """Token strings + IDs. Exact for open models (if tokenizer loads), tiktoken approximation otherwise."""
    repo = OPEN_TOKENIZERS.get(model_id)
    if repo:
        try:
            import os
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo, token=os.getenv("HF_TOKEN"))
            ids = tok.encode(text, add_special_tokens=False)
            return {"tokens": [tok.decode([i]) for i in ids], "ids": ids, "approximate": False, "tokenizer": repo}
        except Exception as e:  # gated/offline: fall through to approximation
            note = f"{repo} unavailable ({type(e).__name__}); "
    else:
        note = ""
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text)
    return {"tokens": [enc.decode([i]) for i in ids], "ids": ids, "approximate": True,
            "tokenizer": note + "tiktoken o200k_base (approximation)"}


STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that these
those there here i you we they he she me my our your their them his her not no so than then too very can could would should
will just do does did done have has had into about over under what which who whom whose when where why how all any each
more most other some such only own same s t don now also yet still""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2}


def echo_score(answer: str, prompt: str) -> float:
    """Q3 (priming vs state): share of the answer's distinct content words that were already in the prompt.
    High echo = the model is recycling the induction's vocabulary rather than saying anything new."""
    a = content_words(answer)
    return 0.0 if not a else len(a & content_words(prompt)) / len(a)
