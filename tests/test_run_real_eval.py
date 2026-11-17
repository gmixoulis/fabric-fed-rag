"""Offline tests for scripts/run_real_eval.py scoring and report building.

Temporary SQLite file plus a monkeypatched generator returning fixed SQL; no
Docker, no Ollama, no snapshot data. Rows here are invented, not client data.
"""

from __future__ import annotations

import sqlite3

import pytest

from scripts import run_real_eval as ev


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(ev.SCHEMA_DDL)
    conn.executemany("INSERT INTO TRDR VALUES (?, ?, ?, ?, ?, ?)", [
        (1, 101, "C001", "Alpha Retail SA", 13, 1),
        (1, 102, "C002", "Beta Commerce Ltd", 13, 1),
    ])
    conn.execute("INSERT INTO MTRL VALUES (1, 501, 'G01', 'Industrial Sensor', 51, 1)")
    conn.execute("INSERT INTO FINDOC VALUES (1, 1001, 'INV-001', '2024-03-15', 1, 1351, 13, 101, 500.0, 120.0, 620.0, 0)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def conn(snapshot):
    connection, is_snapshot = ev.open_eval_db(snapshot)
    assert is_snapshot
    yield connection
    connection.close()


def case(sql, group="plain", cid="t01"):
    return {"id": cid, "question": "offline test question", "sql": sql, "group": group}


def run(conn, monkeypatch, cases, gen_sql, chunks=None):
    monkeypatch.setattr(ev, "ollama_generate_sql", lambda q, c, model="qwen3.5:4b": gen_sql)
    return ev.evaluate_condition(conn, cases, "offline", lambda q: chunks or [])


def test_correct_sql_passes(conn, monkeypatch):
    summary = run(conn, monkeypatch, [case("SELECT COUNT(*) FROM TRDR")], "SELECT COUNT(*) FROM TRDR")
    assert summary["passed"] == 1
    assert summary["cases"][0]["status"] == "PASS"


def test_wrong_sql_fails(conn, monkeypatch):
    summary = run(conn, monkeypatch, [case("SELECT COUNT(*) FROM TRDR")], "SELECT COUNT(*) FROM MTRL")
    assert summary["passed"] == 0
    # No result rows may leak into the status (public repo).
    assert summary["cases"][0]["status"] == "ROW_MISMATCH"


def test_empty_vs_empty_different_sql_fails(conn, monkeypatch):
    gold = case("SELECT NAME FROM TRDR WHERE TRDR = -1")
    summary = run(conn, monkeypatch, [gold], "SELECT NAME FROM TRDR WHERE TRDR = -2")
    assert summary["passed"] == 0
    assert summary["cases"][0]["status"] == "EMPTY_MISMATCH"
    # Whitespace/case-equivalent SQL on two empty results does pass.
    summary = run(conn, monkeypatch, [gold], "select  name FROM trdr WHERE TRDR = -1;")
    assert summary["passed"] == 1


def test_refusal_case_passes_on_cannot_answer(conn, monkeypatch):
    refusal = case("CANNOT_ANSWER", group="refuse")
    summary = run(conn, monkeypatch, [refusal], "CANNOT_ANSWER")
    assert summary["passed"] == 1
    summary = run(conn, monkeypatch, [refusal], "SELECT COUNT(*) FROM TRDR")
    assert summary["passed"] == 0


def test_broken_gold_counted_as_gold_error(conn, monkeypatch):
    summary = run(conn, monkeypatch, [case("SELECT COUNT(*) FROM NO_SUCH_TABLE")], "SELECT 1")
    assert summary["gold_error"] == 1
    assert summary["passed"] == 0
    assert summary["cases"][0]["status"].startswith("GOLD_ERROR")


def test_guard_still_blocks_unverified_code(conn, monkeypatch):
    gold = case("SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1351", group="code")
    summary = run(conn, monkeypatch, [gold], "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1351")
    assert summary["passed"] == 0
    assert summary["cases"][0]["status"].startswith("GUARD_BLOCKED")
    chunks = [{"text": "FINDOC.SOSOURCE 1351", "ledger_hash_verified": True}]
    summary = run(conn, monkeypatch, [gold], "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1351", chunks)
    assert summary["passed"] == 1


def test_open_eval_db_readonly_and_toy_fallback(snapshot, tmp_path):
    conn, is_snapshot = ev.open_eval_db(snapshot)
    assert is_snapshot
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE intruder (x)")
    conn.close()
    conn, is_snapshot = ev.open_eval_db(tmp_path / "missing.sqlite")
    assert not is_snapshot
    assert conn.execute("SELECT COUNT(*) FROM TRDR").fetchone()[0] > 0
    conn.close()


def test_report_header_has_actual_case_count(conn, monkeypatch):
    cases = [
        case("SELECT COUNT(*) FROM TRDR", group="plain", cid="t01"),
        case("SELECT COUNT(*) FROM MTRL", group="code", cid="t02"),
        case("CANNOT_ANSWER", group="refuse", cid="t03"),
    ]
    summary = run(conn, monkeypatch, cases, "CANNOT_ANSWER")
    eval_summary = {key: summary for key in
                    ("no_rag", "single_shard_node1", "decentralized_50_50_rag")}
    db_note = "**WARNING: toy database, numbers not comparable**"
    lines = ev.build_eval_report_lines(cases, eval_summary, db_note)
    header = next(line for line in lines if line.startswith("## 2."))
    assert "3 Cases" in header
    assert db_note in lines
    table_header = next(line for line in lines if line.startswith("| Experimental Condition"))
    for group in ("plain", "code", "refuse"):
        assert group in table_header
    assert "hp01" not in "\n".join(lines)
    per_case = [line for line in lines if line.startswith("| `t0")]
    assert len(per_case) == 3
