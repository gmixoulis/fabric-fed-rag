"""Offline tests for the evaluator's repair, lenient and sql_in_train scores.

Each test uses a temporary SQLite file, a fake generator callable and a fake
tokenizer; no torch model. The real ``softone_rag`` (from the softone-rag
checkout) is used when it imports; otherwise a minimal stand-in with the same
surface (``graph.build_prompt/extract_sql/MAX_REPAIRS``, ``guard.validate_sql``,
``db.query``, ``evaluate.results_equal/sql_equivalent``) is installed for the test
only, so the evaluator logic is still exercised by the offline gate.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_HW_TRAINING = _HERE.parent / "fedrag" / "hw_training"
if str(_HW_TRAINING) not in sys.path:
    sys.path.insert(0, str(_HW_TRAINING))

import evaluate_adapter  # noqa: E402
import softone_training as common  # noqa: E402

SOFTONE_SRC = Path("/home/gmichoulis/softone-rag/src")
REPAIR_PHRASE = "Your previous SQL was rejected with this error: "


def _real_softone() -> bool:
    if SOFTONE_SRC.is_dir() and str(SOFTONE_SRC) not in sys.path:
        sys.path.insert(0, str(SOFTONE_SRC))
    try:
        from softone_rag import db, evaluate, graph, guard, llm, real_catalog  # noqa: F401
    except ImportError:
        return False
    return True


def _fake_softone(monkeypatch) -> None:
    package = types.ModuleType("softone_rag")
    package.__path__ = []

    class DatabaseError(Exception):
        pass

    def query(path, sql):
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cursor = connection.execute(sql)
            return [d[0] for d in cursor.description or []], cursor.fetchall()
        except sqlite3.Error as error:
            raise DatabaseError(f"query refused: {error}") from error
        finally:
            connection.close()

    def validate_sql(sql, allowlist, unverified_codes=frozenset(), verified_codes=None):
        tables = {name.upper() for name in re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", sql, re.I)}
        if not sql.strip().upper().startswith("SELECT"):
            return types.SimpleNamespace(ok=False, sql="", reason="only a single SELECT is allowed")
        for table in sorted(tables):
            if table not in allowlist:
                return types.SimpleNamespace(
                    ok=False, sql="", reason=f"table {table} is not in the allow-list")
        capped = sql if re.search(r"\bLIMIT\b", sql, re.I) else sql + " LIMIT 200"
        return types.SimpleNamespace(ok=True, sql=capped, reason="")

    def norm(value):
        if value is None:
            return ("z", 0.0)
        if isinstance(value, (int, float)):
            return ("n", round(float(value), 6))
        return ("s", str(value))

    def results_equal(rows_a, rows_b, order_sensitive):
        keys_a = [tuple(norm(v) for v in row) for row in rows_a]
        keys_b = [tuple(norm(v) for v in row) for row in rows_b]
        return keys_a == keys_b if order_sensitive else sorted(keys_a) == sorted(keys_b)

    def extract_sql(text):
        match = re.search(r"\bSELECT\b.*", text, re.I | re.S)
        return match.group(0).strip().rstrip(";") if match else None

    def build_prompt(question, tables, error=None, data_catalog=None, memories=None):
        lines = ["Tables: " + ", ".join(sorted(tables))]
        if error:
            lines.append(REPAIR_PHRASE + error + ". Return only the corrected SQL.")
        return "\n\n".join(lines + ["Question:\n" + question])

    modules = {
        "db": {"DatabaseError": DatabaseError, "query": query},
        "guard": {"validate_sql": validate_sql},
        "graph": {"MAX_REPAIRS": 2, "extract_sql": extract_sql, "build_prompt": build_prompt},
        "evaluate": {"results_equal": results_equal,
                     "sql_equivalent": lambda a, b: a.strip() == b.strip()},
        "real_catalog": {"TABLES": {"TRDR": ""}, "ALLOWED_TABLES": frozenset({"TRDR"}),
                         "UNVERIFIED_CODES": frozenset({"SODTYPE"})},
        "llm": {"CANNOT_ANSWER": "CANNOT_ANSWER"},
    }
    monkeypatch.setitem(sys.modules, "softone_rag", package)
    for name, attributes in modules.items():
        module = types.ModuleType(f"softone_rag.{name}")
        module.__dict__.update(attributes)
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, f"softone_rag.{name}", module)


@pytest.fixture
def softone(monkeypatch):
    if not _real_softone():
        _fake_softone(monkeypatch)


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE TRDR (TRDR INTEGER, NAME TEXT, AMOUNT REAL)")
    connection.executemany("INSERT INTO TRDR VALUES (?, ?, ?)",
                           [(1, "Alpha", 10.0), (2, "Beta", 20.0), (3, "Gamma", 21.5)])
    connection.commit()
    connection.close()
    return path


class FakeTokenizer:
    def apply_chat_template(self, messages, **_):
        return messages[0]["content"]


class FakeModel:
    def eval(self):
        return self


class ScriptedGenerator:
    """Replies from a fixed list, in order, and records every prompt it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _evaluate(cases, snapshot, generator, repair=True):
    return common.evaluate_cases(FakeModel(), FakeTokenizer(), cases, snapshot, "cpu",
                                 repair=repair, generate=generator)


def test_bad_then_good_sql_passes_with_repair_only(softone, snapshot):
    case = {"id": "c1", "question": "How many traders?",
            "sql": "SELECT COUNT(*) FROM TRDR"}
    generator = ScriptedGenerator(["SELECT COUNT(*) FROM NO_SUCH_TABLE",
                                   "SELECT COUNT(*) FROM TRDR"])
    [record] = _evaluate([case], snapshot, generator)
    assert record["passed"] is False
    assert record["passed_lenient"] is False
    assert record["passed_repair"] is True
    assert record["repairs"] == 1
    assert len(generator.prompts) == 2
    assert REPAIR_PHRASE not in generator.prompts[0]
    assert REPAIR_PHRASE in generator.prompts[1]


def test_database_error_is_repaired_and_capped(softone, snapshot):
    case = {"id": "c1", "question": "How many traders?",
            "sql": "SELECT COUNT(*) FROM TRDR"}
    bad = "SELECT NO_SUCH_COLUMN FROM TRDR"
    generator = ScriptedGenerator([bad, bad, bad])
    [record] = _evaluate([case], snapshot, generator)
    assert (record["passed"], record["passed_repair"]) == (False, False)
    assert record["repairs"] == 2  # graph.MAX_REPAIRS
    assert "no such column" in generator.prompts[-1]


def test_no_repair_makes_one_generation_and_omits_the_score(softone, snapshot):
    case = {"id": "c1", "question": "How many traders?",
            "sql": "SELECT COUNT(*) FROM TRDR"}
    generator = ScriptedGenerator(["SELECT COUNT(*) FROM NO_SUCH_TABLE"])
    [record] = _evaluate([case], snapshot, generator, repair=False)
    assert "passed_repair" not in record
    assert len(generator.prompts) == 1
    assert "passed_repair" not in common.summarise_records([record])


def test_unrounded_average_passes_lenient_only(softone, snapshot):
    case = {"id": "c1", "question": "Average amount?",
            "sql": "SELECT ROUND(AVG(AMOUNT), 2) FROM TRDR"}
    generator = ScriptedGenerator(["SELECT AVG(AMOUNT) FROM TRDR"])
    [record] = _evaluate([case], snapshot, generator)
    assert record["passed"] is False
    assert record["passed_repair"] is False
    assert record["passed_lenient"] is True


def test_extra_column_answer_passes_lenient_only(softone, snapshot):
    case = {"id": "c1", "question": "Top trader by amount?",
            "sql": "SELECT AMOUNT FROM TRDR ORDER BY AMOUNT DESC LIMIT 2"}
    generator = ScriptedGenerator(
        ["SELECT NAME, AMOUNT FROM TRDR ORDER BY AMOUNT DESC LIMIT 2"])
    [record] = _evaluate([case], snapshot, generator)
    assert record["passed"] is False
    assert record["passed_lenient"] is True


def test_lenient_still_needs_the_gold_values():
    gold = [(10.0,), (20.0,)]
    assert not common.results_match_lenient([("Alpha", 10.0), ("Beta", 21.0)], gold, True)
    assert not common.results_match_lenient([("Alpha", 10.0)], gold, True)
    assert not common.results_match_lenient([("Beta", 20.0), ("Alpha", 10.0)], gold, True)
    assert common.results_match_lenient([("Beta", 20.0), ("Alpha", 10.0)], gold, False)


def test_sql_in_train_split_adds_up_to_total():
    records = [
        {"id": "a", "passed": True, "passed_repair": True, "passed_lenient": True,
         "sql_in_train": True},
        {"id": "b", "passed": False, "passed_repair": True, "passed_lenient": False,
         "sql_in_train": True},
        {"id": "c", "passed": False, "passed_repair": False, "passed_lenient": True,
         "sql_in_train": False},
        {"id": "d", "passed": True, "passed_repair": True, "passed_lenient": True,
         "sql_in_train": False},
        {"id": "e", "passed": False, "passed_repair": False, "passed_lenient": False,
         "sql_in_train": False},
    ]
    summary = common.summarise_records(records)
    split = summary["by_sql_in_train"]
    assert set(split) == {"true", "false"}
    for key in ("cases", "passed", "passed_repair", "passed_lenient"):
        assert split["true"][key] + split["false"][key] == summary[key]
    assert summary["cases"] == 5
    assert (summary["passed"], summary["passed_repair"], summary["passed_lenient"]) == (2, 3, 3)
    assert split["true"] == {"passed": 1, "cases": 2, "passed_repair": 2, "passed_lenient": 1}


def test_old_records_summarise_as_before():
    summary = common.summarise_records([{"id": "a", "passed": True, "group": "g"}])
    assert summary["by_group"] == {"g": {"passed": 1, "cases": 1}}
    assert "passed_repair" not in summary and "passed_lenient" not in summary
    assert summary["by_sql_in_train"] == {"unknown": {"passed": 1, "cases": 1}}


def test_per_case_losses_average_to_reported_mean(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    fake_torch.no_grad = NoGrad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    losses = {"a": 0.25, "b": 1.5, "c": 2.0}
    monkeypatch.setattr(common, "prompt_text", lambda question, facts=None: question)
    monkeypatch.setattr(evaluate_adapter.client, "encode_example",
                        lambda tokenizer, record, max_length: record["id"])
    monkeypatch.setattr(evaluate_adapter.client, "supervised_loss",
                        lambda model, batch, device, torch: types.SimpleNamespace(
                            item=lambda: losses[batch]))
    cases = [{"id": key, "question": "q", "sql": "SELECT 1"} for key in losses]
    per_case = evaluate_adapter.heldout_case_losses(FakeModel(), None, cases, "cpu",
                                                    2304, None)
    assert per_case == losses
    mean = evaluate_adapter.heldout_loss(FakeModel(), None, cases, "cpu", 2304, None)
    assert mean == pytest.approx(sum(per_case.values()) / len(per_case))
    assert evaluate_adapter.mean_loss(per_case) == mean


def test_cli_defaults_and_no_repair_flag():
    assert evaluate_adapter.DEFAULT_MAX_LENGTH == 2304
    base = ["--adapter", "a", "--out", "o.json"]
    assert evaluate_adapter.parse_args(base).repair is True
    assert evaluate_adapter.parse_args([*base, "--no-repair"]).repair is False


def test_reuse_base_checks_the_holdout_ids(tmp_path):
    import json

    import pytest

    from fedrag.hw_training import evaluate_adapter as ev

    report = tmp_path / "r.json"
    report.write_text(json.dumps({"base": {"loss": 0.3}, "before": [{"id": "a"}, {"id": "b"}]}))
    summary, before = ev.reuse_base(report, [{"id": "a"}, {"id": "b"}])
    assert summary["loss"] == 0.3 and [r["id"] for r in before] == ["a", "b"]
    with pytest.raises(RuntimeError):
        ev.reuse_base(report, [{"id": "a"}, {"id": "c"}])
