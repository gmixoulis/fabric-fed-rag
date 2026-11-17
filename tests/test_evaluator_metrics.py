"""Offline unit tests for the held-out evaluator metrics and prompt wiring.

No test here needs torch, a GPU, the network or client data. The only optional
dependency is ``softone_rag``, used for the one prompt test that checks the
production prompt carries the taught facts; it is skipped when that package (or
its dependencies) is not installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_HW_TRAINING = _HERE.parent / "fedrag" / "hw_training"
if str(_HW_TRAINING) not in sys.path:
    sys.path.insert(0, str(_HW_TRAINING))

import coordinator  # noqa: E402
import softone_training as common  # noqa: E402


# --------------------------------------------------------------------------- #
# Wilson score interval
# --------------------------------------------------------------------------- #
def test_wilson_interval_bounds():
    assert common.wilson_interval(0, 0) == (0.0, 0.0)
    assert common.wilson_interval(0, 12) == (0.0, 0.2425)
    assert common.wilson_interval(6, 12) == (0.2538, 0.7462)
    assert common.wilson_interval(12, 12) == (0.7575, 1.0)


def test_wilson_interval_hand_checked():
    # 8/12 with z=1.96, computed by hand and confirmed to four places.
    assert common.wilson_interval(8, 12) == (0.3906, 0.8619)


# --------------------------------------------------------------------------- #
# summarise_records / compare_records
# --------------------------------------------------------------------------- #
def test_summarise_records_groups_and_interval():
    records = [
        {"id": "a", "passed": True, "group": "g1"},
        {"id": "b", "passed": False, "group": "g1"},
        {"id": "c", "passed": True, "group": "g2"},
        {"id": "d", "passed": False},
    ]
    summary = common.summarise_records(records)
    assert summary["passed"] == 2
    assert summary["cases"] == 4
    assert summary["interval"] == list(common.wilson_interval(2, 4))
    assert summary["by_group"] == {
        "g1": {"passed": 1, "cases": 2},
        "g2": {"passed": 1, "cases": 1},
        "": {"passed": 0, "cases": 1},
    }


def test_compare_records_gained_and_lost():
    base = [
        {"id": "a", "passed": True},
        {"id": "b", "passed": True},
        {"id": "c", "passed": False},
    ]
    other = [
        {"id": "a", "passed": False},
        {"id": "b", "passed": True},
        {"id": "c", "passed": True},
    ]
    assert common.compare_records(base, other) == {"gained": ["c"], "lost": ["a"]}


# --------------------------------------------------------------------------- #
# Refusal cases
# --------------------------------------------------------------------------- #
def test_is_refusal_case():
    assert common.is_refusal_case({"sql": "CANNOT_ANSWER"})
    assert common.is_refusal_case({"sql": "  CANNOT_ANSWER  "})
    assert common.is_refusal_case({"sql": ""})
    assert common.is_refusal_case({})  # no sql key at all
    assert not common.is_refusal_case({"sql": "SELECT 1"})


# --------------------------------------------------------------------------- #
# load_taught_facts
# --------------------------------------------------------------------------- #
def test_load_taught_facts_adds_kind_in_order(tmp_path):
    path = tmp_path / "facts.jsonl"
    path.write_text(
        json.dumps({"text": "first fact", "table_name": "T"}) + "\n"
        + json.dumps({"text": "second fact", "table_name": "U"}) + "\n",
        encoding="utf-8",
    )
    facts = common.load_taught_facts(path)
    assert [fact["text"] for fact in facts] == ["first fact", "second fact"]
    assert all(fact["kind"] == "fact" for fact in facts)
    assert facts[0]["table_name"] == "T"
    assert facts[1]["table_name"] == "U"


def test_load_taught_facts_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no taught facts"):
        common.load_taught_facts(path)


# --------------------------------------------------------------------------- #
# build_evaluate_command
# --------------------------------------------------------------------------- #
def test_build_evaluate_command_with_two_adapters():
    command = coordinator.build_evaluate_command(
        train_safe="train-safe",
        python="python",
        script="evaluate_adapter.py",
        adapters=["/exp/round-1/global", "/exp/round-2/global"],
        out="/exp/eval.json",
        holdout="/exp/holdout.jsonl",
        train_data="/exp/train.jsonl",
    )
    assert command[:3] == ["train-safe", "--", "python"]
    assert command.count("--adapter") == 2
    first = command.index("--adapter")
    second = command.index("--adapter", first + 1)
    assert command[first + 1] == "/exp/round-1/global"
    assert command[second + 1] == "/exp/round-2/global"
    assert "--out" in command and "--holdout" in command and "--train-data" in command


# --------------------------------------------------------------------------- #
# Production prompt carries the taught facts
# --------------------------------------------------------------------------- #
def test_prompt_text_includes_taught_facts():
    src = "/home/gmichoulis/softone-rag/src"
    if not Path(src).is_dir():
        pytest.skip("softone-rag checkout not available on this host")
    sys.path.insert(0, src)
    pytest.importorskip("softone_rag")
    try:
        import softone_rag.graph  # noqa: F401  (pulls in langgraph and friends)
    except ImportError:
        pytest.skip("softone_rag graph dependencies are not installed here")

    fact_text = "Customers are traders with TRDR.SODTYPE 13."
    prompt = common.prompt_text(
        "How many customers are there?", facts=[{"kind": "fact", "text": fact_text}]
    )
    assert fact_text in prompt
    assert "Known facts" in prompt
