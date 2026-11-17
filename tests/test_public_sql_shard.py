"""Offline tests for scripts/public_sql_shard.py.

No pyarrow, no softone_rag, no network: input is JSONL in ``tmp_path`` and the
prompt builder is replaced with a fake, so ``softone_rag`` is never imported.
"""

from __future__ import annotations

import json
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import public_sql_shard as shard  # noqa: E402

VALID_CONTEXT = (
    "CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER);\n"
    "INSERT INTO employees VALUES (1, 'Alice', 50000);\n"
    "INSERT INTO employees VALUES (2, 'Bob', 60000);"
)

GOOD_ROW = {
    "sql_prompt": "Who earns more than 55000?",
    "sql_context": VALID_CONTEXT,
    "sql": "SELECT name FROM employees WHERE salary > 55000;",
}


def _base_argv(input_path: Path, out_path: Path) -> list[str]:
    return [
        "--input", str(input_path),
        "--out", str(out_path),
        "--source-name", "test-dataset",
        "--source-license", "CC-BY-4.0",
        "--source-url", "https://example.com/data",
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def run_shard(monkeypatch, tmp_path, capsys):
    def _run(rows, *, limit=10, seed=0, max_sql_chars=400, exclude=(),
             fields=("sql_prompt", "sql_context", "sql"), out_name="shard.jsonl"):
        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        out_path = tmp_path / out_name

        def fake_builder(question, tables, data_catalog):
            return f"PROMPT|{question}|{','.join(sorted(tables))}"

        monkeypatch.setattr(shard, "build_prompt", fake_builder)

        argv = _base_argv(input_path, out_path) + [
            "--question-field", fields[0],
            "--context-field", fields[1],
            "--sql-field", fields[2],
            "--max-sql-chars", str(max_sql_chars),
            "--limit", str(limit),
            "--seed", str(seed),
        ]
        for path in exclude:
            argv += ["--exclude", str(path)]
        code = shard.main(argv)
        captured = capsys.readouterr()
        return code, out_path, captured.out

    return _run


def test_good_row_kept(run_shard):
    code, out_path, _ = run_shard([GOOD_ROW], limit=5, seed=0)
    assert code == 0
    records = _read_jsonl(out_path)
    assert len(records) == 1
    record = records[0]
    assert record["id"] == "pub-0"
    assert record["question"] == GOOD_ROW["sql_prompt"]
    assert record["prompt"] == "PROMPT|Who earns more than 55000?|employees"
    assert record["target"] == "SELECT name FROM employees WHERE salary > 55000"
    assert record["source"] == "test-dataset"


def test_rejection_reasons(run_shard):
    rows = [
        {"sql_prompt": "drop q", "sql_context": VALID_CONTEXT,
         "sql": "INSERT INTO employees VALUES (3, 'C', 1);"},
        {"sql_prompt": "broken context", "sql_context": "CREATE TABLE broken (id INTEGER;",
         "sql": "SELECT 1;"},
        {"sql_prompt": "missing table", "sql_context": VALID_CONTEXT,
         "sql": "SELECT * FROM missing;"},
        {"sql_prompt": "no rows", "sql_context": VALID_CONTEXT,
         "sql": "SELECT * FROM employees WHERE 0;"},
        {"sql_prompt": "too long", "sql_context": VALID_CONTEXT,
         "sql": "SELECT name FROM employees WHERE salary > 55000 ORDER BY name;"},
        {"sql_prompt": "", "sql_context": VALID_CONTEXT, "sql": "SELECT 1;"},
    ]
    code, out_path, stdout = run_shard(rows, max_sql_chars=50)
    assert code == 0
    assert _read_jsonl(out_path) == []
    summary = json.loads(stdout.strip())
    assert summary["total"] == 6
    assert summary["selected"] == 0
    for reason in ("not_select", "bad_context", "sql_error", "empty_result",
                   "sql_too_long", "empty_fields"):
        assert summary[reason] == 1, reason


def test_exclusion_file(run_shard, tmp_path):
    exclude_path = tmp_path / "exclude.jsonl"
    exclude_path.write_text(
        json.dumps({"question": "  Who   Earns   more than 55000?  "}) + "\n",
        encoding="utf-8",
    )
    code, out_path, stdout = run_shard([GOOD_ROW], exclude=[exclude_path])
    assert code == 0
    assert _read_jsonl(out_path) == []
    summary = json.loads(stdout.strip())
    assert summary["excluded"] == 1
    assert summary["selected"] == 0


def test_deduplication(run_shard):
    rows = [
        {"sql_prompt": "How many employees?", "sql_context": VALID_CONTEXT,
         "sql": "SELECT COUNT(*) FROM employees;"},
        {"sql_prompt": "  how many   EMPLOYEES? ", "sql_context": VALID_CONTEXT,
         "sql": "SELECT COUNT(*) FROM employees;"},
        {"sql_prompt": "What is Bob's salary?", "sql_context": VALID_CONTEXT,
         "sql": "SELECT salary FROM employees WHERE name = 'Bob';"},
    ]
    code, out_path, stdout = run_shard(rows, limit=10)
    assert code == 0
    records = _read_jsonl(out_path)
    assert len(records) == 2
    assert {record["question"] for record in records} == {
        "How many employees?", "What is Bob's salary?",
    }
    summary = json.loads(stdout.strip())
    assert summary["duplicate"] == 1
    assert summary["selected"] == 2


def test_round_robin_selection():
    records = [{"complexity": "easy", "q": f"easy {i}"} for i in range(4)]
    records += [{"complexity": "hard", "q": f"hard {i}"} for i in range(4)]
    selected = shard.select_records(records, 4, seed=0, has_complexity=True)
    counts = Counter(record["complexity"] for record in selected)
    assert counts == {"easy": 2, "hard": 2}


def test_complexity_column_end_to_end(run_shard):
    def make_row(label, complexity):
        return {
            "sql_prompt": f"{label} question",
            "sql_context": VALID_CONTEXT,
            "sql": "SELECT name FROM employees WHERE salary > 55000;",
            "sql_complexity": complexity,
        }

    rows = [make_row(f"easy {i}", "easy") for i in range(4)]
    rows += [make_row(f"hard {i}", "hard") for i in range(4)]
    code, out_path, _ = run_shard(rows, limit=4, seed=0)
    assert code == 0
    records = _read_jsonl(out_path)
    assert len(records) == 4
    easy = [r for r in records if r["question"].startswith("easy")]
    hard = [r for r in records if r["question"].startswith("hard")]
    assert len(easy) == 2
    assert len(hard) == 2


def test_alternate_field_names(run_shard):
    row = {
        "question": "Who earns more than 55000?",
        "context": VALID_CONTEXT,
        "answer": "SELECT name FROM employees WHERE salary > 55000;",
    }
    code, out_path, _ = run_shard([row], fields=("question", "context", "answer"))
    assert code == 0
    records = _read_jsonl(out_path)
    assert len(records) == 1
    assert records[0]["question"] == row["question"]
    assert records[0]["target"] == "SELECT name FROM employees WHERE salary > 55000"


def test_refuse_inside_repository(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(GOOD_ROW) + "\n", encoding="utf-8")
    inside = shard.REPO_ROOT / "public_sql_shard_never_created.jsonl"
    with pytest.raises(shard.ShardError, match="inside the repository"):
        shard.main(_base_argv(input_path, inside))
    assert not inside.exists()


def test_refuse_overwrite(tmp_path):
    out_path = tmp_path / "shard.jsonl"
    out_path.write_text("existing\n", encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(GOOD_ROW) + "\n", encoding="utf-8")
    with pytest.raises(shard.ShardError, match="refusing to overwrite"):
        shard.main(_base_argv(input_path, out_path))


def test_source_json_contents(run_shard, tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(GOOD_ROW) + "\n", encoding="utf-8")
    code, out_path, _ = run_shard([GOOD_ROW], seed=42)
    assert code == 0
    source = json.loads((Path(out_path.parent) / (out_path.name + ".source.json")).read_text())
    assert source["source_name"] == "test-dataset"
    assert source["source_license"] == "CC-BY-4.0"
    assert source["source_url"] == "https://example.com/data"
    assert source["seed"] == 42
    assert source["input_sha256"] == shard.sha256_file(input_path)
    assert source["counts"]["selected"] == 1
    assert source["counts"]["total"] == 1


def test_file_modes(run_shard):
    code, out_path, _ = run_shard([GOOD_ROW])
    assert code == 0
    source_path = out_path.parent / (out_path.name + ".source.json")
    assert stat.S_IMODE(out_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(source_path.stat().st_mode) == 0o600
