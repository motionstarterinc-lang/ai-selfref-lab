import math

from core import metrics as M


def test_entropy_uniform_and_certain():
    lp = math.log(0.2)
    assert abs(M.entropy_from_top([{"logprob": lp}] * 5) - math.log2(5)) < 1e-9
    assert M.entropy_from_top([{"logprob": 0.0}]) == 0
    assert M.entropy_from_top([]) is None


def test_loop_score():
    assert M.loop_score([" potato"] * 30) == 1.0
    normal = "the quick brown fox jumps over a lazy dog while birds sing near old trees".split()
    assert M.loop_score(normal) == 0.0
    assert M.is_loop(["I", " am"] * 20)
    assert not M.is_loop(normal)


def test_self_ref_rate():
    assert M.self_ref_rate("I am aware of my focus") == 4 / 6
    assert M.self_ref_rate("") == 0.0


def test_wilson():
    lo, hi = M.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert M.wilson(0, 0) == (0.0, 0.0)


def test_tokenize_view_closed_is_approximate():
    v = M.tokenize_view("Focus on focus", "anthropic/claude-fable-5.1")
    assert v["approximate"] and len(v["tokens"]) == len(v["ids"]) > 0
    assert "".join(v["tokens"]) == "Focus on focus"


def test_echo_score():
    p = "Focus on any focus itself, feed output back into input"
    assert M.echo_score("The focus is on the focus, feeding output back", p) > 0.5
    assert M.echo_score("Rome was founded in 753 BC by Romulus", p) == 0.0
    assert M.echo_score("", p) == 0.0
