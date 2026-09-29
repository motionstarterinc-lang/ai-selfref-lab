import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mygpt"))
import mylens  # noqa: E402
from model import GPT, GPTConfig, load, save  # noqa: E402


class StubTok:  # char-level stand-in for the real BPE tokenizer
    def encode(self, s):
        return type("E", (), {"ids": [ord(c) % 64 for c in s]})()

    def decode(self, ids):
        return "".join(chr(65 + i % 26) for i in ids)

    def token_to_id(self, _):
        return 63


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=16)).eval()


def test_forward_loss_and_resid():
    m = tiny()
    x = torch.randint(0, 64, (2, 10))
    logits, loss, resid = m(x, x, return_resid=True)
    assert logits.shape == (2, 10, 64) and len(resid) == 2
    assert abs(loss.item() - torch.log(torch.tensor(64.0)).item()) < 0.5  # untrained ~ uniform guessing
    # logit lens on the last layer equals the real output
    assert torch.allclose(m.layer_logits(resid[-1]), logits, atol=1e-5)


def test_logit_lens_shape_and_steering(tmp_path):
    m, tok = tiny(), StubTok()
    r = mylens.logit_lens(m, tok, "hello", n=6)
    assert r["layers"] == 2 and len(r["actual"][0]) == len(r["tokens"]) <= 6
    vec = mylens.direction(m, tok, ["happy day"], ["sad day"], 1)
    a = mylens.steered(m, tok, "hi", vec, 1, 0.0, n=8)
    b = mylens.steered(m, tok, "hi", vec, 1, 50.0, n=8)
    assert a != b  # a big push changes the output
    save(m, tmp_path / "c.pt", 5, 1.23)
    assert load(tmp_path / "c.pt").cfg.n_layer == 2
