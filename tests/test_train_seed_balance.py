"""Offline balance checks for eval/softone_train_seed.jsonl.

No snapshot and no softone_rag: seed questions must stay disjoint from the
holdout, the appended lines (after the original 124) must not reuse holdout
gold SQL verbatim, seed questions must be unique, and per-year/per-month SQL
must use SUBSTR/strftime rather than YEAR()/MONTH().
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_EVAL = Path(__file__).resolve().parent.parent / "eval"
ORIGINAL_SEED_LINES = 124


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _seed() -> list[dict]:
    return _read_jsonl(_EVAL / "softone_train_seed.jsonl")


def _holdout() -> list[dict]:
    return _read_jsonl(_EVAL / "softone_holdout.jsonl")


def test_seed_questions_disjoint_from_holdout():
    holdout_questions = {_norm(case["question"]) for case in _holdout()}
    leaked = [record["question"] for record in _seed()
              if _norm(record["question"]) in holdout_questions]
    assert leaked == []


def test_appended_sql_not_identical_to_holdout_gold():
    gold = {case["sql"].strip() for case in _holdout()}
    added = _seed()[ORIGINAL_SEED_LINES:]
    assert added, "expected appended seed lines beyond the original 124"
    reused = [record["question"] for record in added if record["sql"].strip() in gold]
    assert reused == []


def test_seed_questions_unique():
    counts = Counter(_norm(record["question"]) for record in _seed())
    assert all(count == 1 for count in counts.values())


def test_no_year_or_month_functions():
    pattern = re.compile(r"\b(YEAR|MONTH)\s*\(", re.IGNORECASE)
    offenders = [record["question"] for record in _seed() if pattern.search(record["sql"])]
    assert offenders == []
