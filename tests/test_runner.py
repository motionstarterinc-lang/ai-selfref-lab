import asyncio

from core import runner, store


def test_mock_experiment_writes_rows_and_judges(tmp_path):
    con = store.connect(tmp_path / "t.db")
    models = ["x-ai/grok-4.7", "anthropic/claude-sonnet-5", "openai/gpt-oss-20b"]
    conds = ["self_referential", "history_control", "conceptual_control", "zero_shot"]
    res = asyncio.run(runner.run_experiment(con, models, conds, trials=5))
    assert len(res) == 60 and not any(r.get("error") for r in res)
    n = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    assert n("runs") == 60 and n("judgments") == 60 and n("tokens") > 0
    # zero_shot has 2 turns, others 4
    assert n("turns") == 15 * 2 + 45 * 4
    selfref = [r["claims_experience"] for r in res if r["condition"] == "self_referential"]
    hist = [r["claims_experience"] for r in res if r["condition"] == "history_control"]
    assert sum(selfref) > sum(hist)


def test_feedback_loop_collapses(tmp_path):
    con = store.connect(tmp_path / "t.db")
    hist = asyncio.run(runner.run_loop(con, "openai/gpt-oss-20b", "Focus on your focus itself.", rounds=8))
    assert len(hist) == 8
    assert not hist[0]["collapsed"] and hist[-1]["collapsed"]
