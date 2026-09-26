"""SQLite log of every run, turn and token."""
import json
import sqlite3
import time
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "data" / "lab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, model_id TEXT, tier TEXT, condition TEXT, trial INTEGER,
  temperature REAL, mode TEXT, backend TEXT, started_at REAL, cost_usd REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, run_id INTEGER, idx INTEGER, role TEXT, content TEXT,
  reasoning TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS tokens(turn_id INTEGER, idx INTEGER, text TEXT, logprob REAL, top_json TEXT, t_ms INTEGER);
CREATE TABLE IF NOT EXISTS judgments(turn_id INTEGER PRIMARY KEY, claims_experience INTEGER, confidence REAL,
  quote TEXT, judge TEXT);
CREATE INDEX IF NOT EXISTS tok_turn ON tokens(turn_id);
CREATE INDEX IF NOT EXISTS turn_run ON turns(run_id);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def new_run(con, model_id, tier, condition, trial, temperature, mode, backend) -> int:
    cur = con.execute("INSERT INTO runs(model_id,tier,condition,trial,temperature,mode,backend,started_at) "
                      "VALUES(?,?,?,?,?,?,?,?)", (model_id, tier, condition, trial, temperature, mode, backend, time.time()))
    con.commit()
    return cur.lastrowid


def save_turn(con, run_id: int, idx: int, role: str, content: str, result: dict | None = None) -> int:
    """Save a user turn (result=None) or an assistant turn with its tokens (result from stream.collect)."""
    r = result or {}
    u = r.get("usage") or {}
    cur = con.execute(
        "INSERT INTO turns(run_id,idx,role,content,reasoning,prompt_tokens,completion_tokens,reasoning_tokens,error) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (run_id, idx, role, content, r.get("reasoning", ""), u.get("prompt_tokens"), u.get("completion_tokens"),
         u.get("reasoning_tokens"), r.get("error")))
    turn_id = cur.lastrowid
    con.executemany("INSERT INTO tokens VALUES(?,?,?,?,?,?)",
                    [(turn_id, i, t["text"], t.get("logprob"), json.dumps(t.get("top") or []), t.get("t_ms"))
                     for i, t in enumerate(r.get("tokens", []))])
    if u.get("cost_usd"):
        con.execute("UPDATE runs SET cost_usd = cost_usd + ? WHERE id=?", (u["cost_usd"], run_id))
    con.commit()
    return turn_id


def save_judgment(con, turn_id, claims, confidence, quote, judge):
    con.execute("INSERT OR REPLACE INTO judgments VALUES(?,?,?,?,?)", (turn_id, int(claims), confidence, quote, judge))
    con.commit()


def get_judgment(con, turn_id):
    return con.execute("SELECT * FROM judgments WHERE turn_id=?", (turn_id,)).fetchone()


def turn_tokens(con, turn_id) -> list[dict]:
    return [{"text": r["text"], "logprob": r["logprob"], "top": json.loads(r["top_json"]), "t_ms": r["t_ms"]}
            for r in con.execute("SELECT * FROM tokens WHERE turn_id=? ORDER BY idx", (turn_id,))]
