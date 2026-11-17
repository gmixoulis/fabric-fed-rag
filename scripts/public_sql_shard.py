#!/usr/bin/env python3
"""Turn a public text-to-SQL dataset into a training-only shard.

The shard is written OUTSIDE the repository (public data is auxiliary training
material only) as JSONL records that ``fedrag/hw_training/client.py`` accepts via
``read_shard``: each record carries ``prompt`` and ``target`` (plus the question
and the source name), matching the ``{id, question, prompt, target}`` shape that
``softone_training.prepare_texts`` produces.

Input is either a parquet file (``gretelai/synthetic_text_to_sql`` layout:
``sql_prompt``, ``sql_context``, ``sql``) or a JSONL file, with the column names
configurable for ``b-mc2/sql-create-context`` (``question``, ``context``,
``answer``). ``pyarrow`` and ``softone_rag`` are both imported lazily so the
module loads without either, and the tests can inject a fake prompt builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

# The repository this script lives in; the output must stay outside it.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The SQL execution budget: a runaway query is aborted by the sqlite3 progress
# handler once wall-clock time passes this many seconds.
QUERY_TIME_BUDGET_SECONDS = 1.0
_PROGRESS_HANDLER_INTERVAL = 100

# The guard's "starts with SELECT or WITH" test. WITH is accepted because a
# leading CTE is still a read-only SELECT.
_SELECT_START = re.compile(r"(?i)^\s*(SELECT|WITH)\b")

REJECTION_REASONS = (
    "excluded",
    "empty_fields",
    "sql_too_long",
    "not_select",
    "bad_context",
    "sql_error",
    "empty_result",
)


class ShardError(RuntimeError):
    """A user-facing error reported by the entry point without a traceback."""


def normalised_question(text: str) -> str:
    """Whitespace-collapsed, case-folded question text, for overlap checks."""
    return " ".join(str(text or "").split()).casefold()


def normalise_target(sql: str) -> str:
    """The target SQL: whitespace collapsed and one trailing semicolon removed."""
    text = " ".join(str(sql or "").split())
    return text[:-1] if text.endswith(";") else text


def is_single_read_only_select(sql: str) -> bool:
    """True when ``sql`` is a single statement that starts with SELECT or WITH.

    This is a lightweight guard, not the full ``softone_rag.guard``: it rejects a
    second statement (any remaining semicolon after a trailing one is stripped)
    and anything that does not start with SELECT or WITH. It may over-reject
    semicolons inside string literals, which is the conservative choice for
    auxiliary training data.
    """
    text = str(sql or "").strip()
    if not _SELECT_START.match(text):
        return False
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return ";" not in text


def _field(row: dict, name: str) -> str:
    """A row field as a string; missing/None become the empty string."""
    value = row.get(name)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _read_jsonl(path: Path, what: str) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ShardError(f"{path}:{number} in {what} is not valid JSON: {error}") from error
    return records


def read_input(path: Path) -> list[dict]:
    """Read a ``.jsonl`` or ``.parquet`` file into a list of row dicts."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path, "input")
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ShardError(
                "reading a .parquet input needs pyarrow; install it or pass a .jsonl file"
            ) from error
        return pq.read_table(str(path)).to_pylist()
    raise ShardError(f"unsupported input format {path.suffix!r}; use .parquet or .jsonl")


def read_exclusions(path: Path) -> set[str]:
    """The set of normalised questions held by one JSONL exclusion file."""
    questions: set[str] = set()
    for number, record in enumerate(_read_jsonl(path, "exclusion file"), 1):
        if record.get("question") is None:
            raise ShardError(f"{path}:{number} in exclusion file has no 'question' field")
        questions.add(normalised_question(record["question"]))
    return questions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RowCatalog:
    """The duck-typed catalog object ``build_prompt`` reads for one row.

    ``build_prompt`` touches only ``UNVERIFIED_CODES`` and ``catalog_text_for``;
    ``ALLOWED_TABLES`` is kept for the guard and for the caller to pass as the
    prompt's table list.
    """

    def __init__(self, create_statements: dict[str, str]):
        self.ALLOWED_TABLES = frozenset(create_statements)
        self.UNVERIFIED_CODES: frozenset[str] = frozenset()
        self._create = create_statements

    def catalog_text_for(self, tables) -> str:
        """The row's CREATE TABLE statements (no INSERT rows), in table order."""
        return "\n\n".join(self._create[name] for name in tables if name in self._create)


def build_prompt(question: str, tables: list[str], data_catalog) -> str:
    """Build the production prompt, importing ``softone_rag`` lazily.

    Kept as a module-level function so the tests can monkeypatch it with a fake
    builder and never import ``softone_rag``.
    """
    try:
        from softone_rag.graph import build_prompt as _build_prompt
    except ImportError as error:
        raise ShardError(
            "softone_rag is not importable; run with its src/ directory on PYTHONPATH "
            "(e.g. PYTHONPATH=/path/to/softone-rag/src)"
        ) from error
    return _build_prompt(question, tables, data_catalog=data_catalog)


def _table_create_statements(conn: sqlite3.Connection) -> dict[str, str]:
    """Table name -> original CREATE TABLE SQL, from sqlite's own catalog."""
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {name: sql for name, sql in rows}


def _progress_handler(deadline: float):
    """A sqlite3 progress handler that aborts once the time budget is spent."""

    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    return handler


def process_row(
    row: dict,
    question_field: str,
    context_field: str,
    sql_field: str,
    max_sql_chars: int,
    excluded_questions: set[str],
    builder,
) -> tuple[str | None, dict | None]:
    """Classify one row: a rejection reason, or (None, record).

    ``record`` carries the fields needed for de-duplication, round-robin selection
    and the final output. Checks run cheapest-first so each row is counted under
    the first reason it fails.
    """
    question = _field(row, question_field)
    sql = _field(row, sql_field)
    context = _field(row, context_field)

    if normalised_question(question) in excluded_questions:
        return "excluded", None

    if not question.strip() or not sql.strip():
        return "empty_fields", None
    if len(sql.strip()) > max_sql_chars:
        return "sql_too_long", None
    if not is_single_read_only_select(sql):
        return "not_select", None

    conn = sqlite3.connect(":memory:")
    try:
        try:
            conn.executescript(context)
        except sqlite3.Error:
            return "bad_context", None

        create = _table_create_statements(conn)
        if not create:
            return "bad_context", None

        deadline = time.monotonic() + QUERY_TIME_BUDGET_SECONDS
        conn.set_progress_handler(_progress_handler(deadline), _PROGRESS_HANDLER_INTERVAL)
        try:
            cursor = conn.execute(sql)
            first = cursor.fetchone()
        except sqlite3.Error:
            return "sql_error", None
    finally:
        conn.close()

    if first is None:
        return "empty_result", None

    catalog = RowCatalog(create)
    tables = sorted(catalog.ALLOWED_TABLES)
    prompt = builder(question, tables, catalog)

    return None, {
        "question": question,
        "prompt": prompt,
        "target": normalise_target(sql),
        "complexity": row.get("sql_complexity"),
    }


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Drop records whose normalised question was already seen; keep the first."""
    seen: set[str] = set()
    unique: list[dict] = []
    duplicates = 0
    for record in records:
        key = normalised_question(record["question"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, duplicates


def select_records(records: list[dict], limit: int, seed: int,
                   has_complexity: bool) -> list[dict]:
    """Choose up to ``limit`` records with ``random.Random(seed)``.

    With a ``sql_complexity`` column the selection round-robins across its values
    (each group shuffled with the seeded RNG first) so the spread stays even.
    """
    rng = random.Random(seed)
    if not has_complexity:
        return rng.sample(records, min(limit, len(records)))

    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record["complexity"]), []).append(record)
    pools = list(groups.values())
    for pool in pools:
        rng.shuffle(pool)

    selected: list[dict] = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for pool in pools:
            if index < len(pool):
                selected.append(pool[index])
                progressed = True
                if len(selected) == limit:
                    return selected
        if not progressed:
            return selected
        index += 1
    return selected


def _open_new(path: Path):
    """Open a file for writing, refusing to overwrite and forcing mode 0600."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    return os.fdopen(os.open(str(path), flags, 0o600), "w", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input .parquet or .jsonl file")
    parser.add_argument("--question-field", default="sql_prompt",
                        help="question column (default: sql_prompt)")
    parser.add_argument("--context-field", default="sql_context",
                        help="schema/context column (default: sql_context)")
    parser.add_argument("--sql-field", default="sql",
                        help="SQL column (default: sql)")
    parser.add_argument("--max-sql-chars", type=int, default=400,
                        help="reject SQL longer than this (default: 400)")
    parser.add_argument("--limit", type=int, default=200,
                        help="rows to keep after filtering (default: 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the row selection (default: 0)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="JSONL file of {question} rows to drop (repeatable)")
    parser.add_argument("--out", required=True,
                        help="output shard path, outside this repository")
    parser.add_argument("--source-name", required=True, help="provenance: dataset name")
    parser.add_argument("--source-license", required=True, help="provenance: licence")
    parser.add_argument("--source-url", required=True, help="provenance: URL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit < 1:
        raise ShardError("--limit must be at least 1")
    if args.max_sql_chars < 1:
        raise ShardError("--max-sql-chars must be at least 1")

    out = Path(args.out).expanduser().resolve()
    if REPO_ROOT in out.parents:
        raise ShardError(f"refusing to write inside the repository: {out}")
    if out.exists():
        raise ShardError(f"refusing to overwrite existing file: {out}")
    source_path = out.parent / (out.name + ".source.json")
    if source_path.exists():
        raise ShardError(f"refusing to overwrite existing file: {source_path}")

    input_path = Path(args.input).expanduser()
    rows = read_input(input_path)

    excluded: set[str] = set()
    for exclude_path in args.exclude:
        excluded.update(read_exclusions(Path(exclude_path).expanduser()))

    has_complexity = any("sql_complexity" in row for row in rows)
    counts = {reason: 0 for reason in REJECTION_REASONS}
    passed: list[dict] = []
    for row in rows:
        reason, record = process_row(
            row, args.question_field, args.context_field, args.sql_field,
            args.max_sql_chars, excluded, build_prompt,
        )
        if reason is not None:
            counts[reason] += 1
        else:
            passed.append(record)

    deduped, duplicates = deduplicate(passed)
    selected = select_records(deduped, args.limit, args.seed, has_complexity)

    out.parent.mkdir(parents=True, exist_ok=True)
    with _open_new(out) as handle:
        for number, record in enumerate(selected):
            record = {
                "id": f"pub-{number}",
                "question": record["question"],
                "prompt": record["prompt"],
                "target": record["target"],
                "source": args.source_name,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = dict(counts)
    summary["duplicate"] = duplicates
    summary["total"] = len(rows)
    summary["selected"] = len(selected)
    with _open_new(source_path) as handle:
        json.dump(
            {
                "source_name": args.source_name,
                "source_license": args.source_license,
                "source_url": args.source_url,
                "input_sha256": sha256_file(input_path),
                "seed": args.seed,
                "counts": summary,
            },
            handle, ensure_ascii=False, indent=2,
        )
        handle.write("\n")

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ShardError, ValueError, FileNotFoundError) as error:
        print(f"public_sql_shard refused: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
