#!/usr/bin/env python3
"""Real-Case End-to-End Evaluation on the 6-container mTLS Raft + Private Kubo IPFS + Sharded RAG Cluster.

Documents are routed to their hash owner (`fedrag.shard_store.ShardRing`), so the corpus
spreads over three shards; the scatter-gather mode queries all reachable nodes.

Uses:
- Real Ollama embeddings (`embeddinggemma:latest`, 768-d), one Ollama container per node
- Real Ollama LLM (`qwen3.5:4b`) for SQL generation over the SoftOne ERP schema
- Real Private Kubo IPFS swarm (`ipfs-node1..3` with `LIBP2P_FORCE_PNET=1`)
- Real SoftOne datasets (`eval/taught_facts.jsonl`, `eval/softone_train_seed.jsonl`, `eval/softone_holdout.jsonl`)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fedrag.node_service import make_client_ssl_context, mtls_request  # noqa: E402
from fedrag.shard_store import ShardRing  # noqa: E402
from scripts.run_demo import connect_kubo_private_swarm, make_minimal_pdf, wait_for_nodes  # noqa: E402

# Same default snapshot as fedrag/hw_training/evaluate_adapter.py.
DEFAULT_SNAPSHOT = Path.home() / ".local/share/softone-chat/snapshots/softone_reputation.sqlite"
REFUSAL_TOKEN = "CANNOT_ANSWER"

SCHEMA_DDL = """
CREATE TABLE TRDR (
    COMPANY INTEGER,
    TRDR INTEGER PRIMARY KEY,
    CODE TEXT,
    NAME TEXT,
    SODTYPE INTEGER,
    ISACTIVE INTEGER
);
CREATE TABLE MTRL (
    COMPANY INTEGER,
    MTRL INTEGER PRIMARY KEY,
    CODE TEXT,
    NAME TEXT,
    SODTYPE INTEGER,
    ISACTIVE INTEGER
);
CREATE TABLE FINDOC (
    COMPANY INTEGER,
    FINDOC INTEGER PRIMARY KEY,
    FINCODE TEXT,
    TRNDATE TEXT,
    BRANCH INTEGER,
    SOSOURCE INTEGER,
    SODTYPE INTEGER,
    TRDR INTEGER,
    NETAMNT REAL,
    VATAMNT REAL,
    SUMAMNT REAL,
    ISCANCEL INTEGER
);
CREATE TABLE MTRLINES (
    COMPANY INTEGER,
    FINDOC INTEGER,
    MTRLINES INTEGER,
    MTRL INTEGER,
    QTY1 REAL,
    PRICE REAL,
    LINEVAL REAL
);
"""

SCHEMA_PROMPT = (
    "You are a read-only SQLite SQL generator for a SoftOne ERP database.\n"
    "Tables and columns:\n"
    "- TRDR(COMPANY, TRDR, CODE, NAME, SODTYPE): Traders.\n"
    "- MTRL(COMPANY, MTRL, CODE, NAME, SODTYPE): Materials, goods, and services.\n"
    "- FINDOC(COMPANY, FINDOC, FINCODE, TRNDATE, BRANCH, SOSOURCE, SODTYPE, TRDR, NETAMNT, VATAMNT, SUMAMNT, ISCANCEL): "
    "Business documents. NETAMNT is net amount; SUMAMNT is total document/payment amount.\n"
    "- MTRLINES(COMPANY, FINDOC, MTRLINES, MTRL, QTY1, PRICE, LINEVAL): Document lines.\n\n"
    "Rules:\n"
    "- Output ONLY a single valid SQLite SELECT statement. No markdown, no explanations.\n"
    "- If a question asks for 'all documents' or 'total documents' (παραστατικά συνολικά) without specifying "
    "sales or purchases, do NOT add any WHERE clause (just SELECT COUNT(*) FROM FINDOC or "
    "SELECT ROUND(SUM(NETAMNT), 2) FROM FINDOC).\n"
    "- For specific document types (sales documents, purchase documents, or supplier payment receipts), "
    "filter FINDOC by a single SOSOURCE = <code> from the Verified RAG Facts below "
    "(never use OR with other SOSOURCE codes, and do not add SODTYPE).\n"
    "- NEVER add ISCANCEL = 0 unless the question explicitly says 'not been cancelled' "
    "(and when it does, use ISCANCEL = 0).\n"
    "- NEVER use GROUP BY unless the question explicitly asks 'for each year'.\n"
    "- When asked for 'net amount' use ROUND(SUM(NETAMNT), 2); when asked for 'total amount of payments' (συνολικό ποσό) "
    "use ROUND(SUM(SUMAMNT), 2); when asked for 'total quantity' use ROUND(SUM(QTY1), 2).\n"
    "- Only when asked 'for each year', use SUBSTR(TRNDATE, 1, 4) AS doc_year, COUNT(*) AS document_count "
    "FROM FINDOC GROUP BY SUBSTR(TRNDATE, 1, 4).\n"
    "- Never guess a numeric code for SODTYPE, SOSOURCE, or ISCANCEL unless stated in the Verified RAG Facts below. "
    "If a question requires an untaught code that is not in the Verified RAG Facts, output CANNOT_ANSWER.\n"
)

UNVERIFIED_CODE_COLUMNS = {
    ("TRDR", "SODTYPE"),
    ("MTRL", "SODTYPE"),
    ("FINDOC", "SOSOURCE"),
    ("FINDOC", "SODTYPE"),
    ("FINDOC", "ISCANCEL"),
}


def build_eval_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_DDL)
    trdr_rows = [
        (1, 101, "C001", "Alpha Retail SA", 13, 1),
        (1, 102, "C002", "Beta Commerce Ltd", 13, 1),
        (2, 103, "C003", "Gamma Stores", 13, 1),
        (1, 201, "S001", "Delta Supplies", 12, 1),
        (2, 202, "S002", "Epsilon Tech", 12, 1),
        (1, 301, "B001", "National Bank Account", 14, 1),
        (1, 401, "T001", "VAT Ledger Account", 16, 1),
    ]
    conn.executemany("INSERT INTO TRDR VALUES (?, ?, ?, ?, ?, ?)", trdr_rows)

    mtrl_rows = [
        (1, 501, "G01", "Industrial Sensor", 51, 1),
        (1, 502, "G02", "Control Valve", 51, 1),
        (2, 503, "G03", "Power Supply", 51, 1),
        (1, 601, "SRV01", "Annual Maintenance Service", 52, 1),
        (2, 602, "SRV02", "On-Site Calibration Service", 52, 1),
        (1, 701, "EXP01", "Electricity Expense", 53, 1),
        (1, 801, "LED01", "VAT Adjustment Line", 61, 1),
    ]
    conn.executemany("INSERT INTO MTRL VALUES (?, ?, ?, ?, ?, ?)", mtrl_rows)

    findoc_rows = [
        (1, 1001, "INV-001", "2024-03-15", 1, 1351, 13, 101, 500.0, 120.0, 620.0, 0),
        (1, 1002, "INV-002", "2024-07-20", 2, 1351, 13, 102, 300.0, 72.0, 372.0, 0),
        (2, 1003, "INV-003", "2025-01-10", 1, 1351, 13, 103, 750.0, 180.0, 930.0, 0),
        (1, 1004, "PUR-001", "2024-05-11", 1, 1253, 12, 201, 400.0, 96.0, 496.0, 0),
        (2, 1005, "PUR-002", "2025-02-18", 3, 1253, 12, 202, 250.0, 60.0, 310.0, 0),
        (1, 1006, "PAY-001", "2025-03-01", 1, 1281, 12, 201, 400.0, 0.0, 400.0, 0),
        (2, 1007, "PAY-002", "2025-04-12", 2, 1281, 12, 202, 150.0, 0.0, 150.0, 0),
    ]
    conn.executemany(
        "INSERT INTO FINDOC VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", findoc_rows
    )

    mtrlines_rows = [
        (1, 1001, 1, 501, 5.0, 100.0, 500.0),
        (1, 1002, 1, 502, 3.0, 100.0, 300.0),
        (2, 1003, 1, 601, 1.5, 500.0, 750.0),
        (1, 1004, 1, 503, 4.0, 100.0, 400.0),
    ]
    conn.executemany("INSERT INTO MTRLINES VALUES (?, ?, ?, ?, ?, ?, ?)", mtrlines_rows)
    conn.commit()
    return conn


def open_eval_db(snapshot: Path) -> tuple[sqlite3.Connection, bool]:
    """Open the real snapshot read-only, or fall back to the toy DB (flagged)."""
    snapshot = Path(snapshot).expanduser()
    if snapshot.is_file():
        return sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True), True
    return build_eval_sqlite(), False


def extract_unlocked_codes(chunks: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Parse verified (TABLE, COLUMN, CODE) triples taught in retrieved hash-verified RAG chunks."""
    unlocked: dict[tuple[str, str], set[str]] = {}
    pattern = re.compile(r"\b(TRDR|MTRL|FINDOC)\.(SODTYPE|SOSOURCE|ISCANCEL)\s+(\d+)\b")
    for chunk in chunks:
        if not chunk.get("ledger_hash_verified", True):
            continue
        for match in pattern.finditer(chunk["text"]):
            table, col, val = match.group(1), match.group(2), match.group(3)
            unlocked.setdefault((table, col), set()).add(val)
    return unlocked


def check_sql_guard(sql: str, unlocked: dict[tuple[str, str], set[str]]) -> tuple[bool, str]:
    """Verify SQL is a single SELECT and only compares code columns against unlocked literals."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        return False, "not a SELECT statement"
    if ";" in cleaned:
        return False, "multiple statements"

    upper_sql = cleaned.upper()
    tables_in_query = [t for t in ("TRDR", "MTRL", "FINDOC", "MTRLINES") if t in upper_sql]
    for col in ("SODTYPE", "SOSOURCE", "ISCANCEL"):
        for match in re.finditer(rf"\b{col}\s*=\s*(\d+)\b", upper_sql):
            literal = match.group(1)
            allowed = False
            for table in tables_in_query:
                if (table, col) in UNVERIFIED_CODE_COLUMNS and literal in unlocked.get(
                    (table, col), set()
                ):
                    allowed = True
            if not allowed:
                return False, f"unverified code {col}={literal} not unlocked by retrieved RAG facts"
    return True, cleaned


def ollama_generate_sql(question: str, retrieved_chunks: list[dict], model: str = "qwen3.5:4b") -> str:
    facts_block = (
        "\n".join(f"- {c['text']}" for c in retrieved_chunks)
        if retrieved_chunks
        else "(No verified RAG facts retrieved.)"
    )
    prompt = f"{SCHEMA_PROMPT}\nVerified RAG Facts:\n{facts_block}\n\nQuestion: {question}\nSQL:"
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "think": False,
                "stream": False,
                "options": {"temperature": 0.0},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode())["message"]["content"].strip()
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def sql_equivalent(a: str, b: str) -> bool:
    """SQL equality after whitespace/case normalisation (for empty-vs-empty)."""
    def norm(sql: str) -> str:
        return " ".join(sql.lower().split()).rstrip(";").strip()

    return norm(a) == norm(b)


def _norm_value(value):
    if value is None:
        return ("z", 0.0)
    if isinstance(value, (int, float)):
        return ("n", round(float(value), 6))
    return ("s", str(value))


def results_equal(actual: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Row equality at 6 decimals; order kept only when the gold SQL orders."""
    keys_a = [tuple(_norm_value(v) for v in row) for row in actual]
    keys_b = [tuple(_norm_value(v) for v in row) for row in gold]
    return keys_a == keys_b if ordered else sorted(keys_a) == sorted(keys_b)


def gold_is_ordered(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", str(sql or ""), re.IGNORECASE))


def score_case(conn: sqlite3.Connection, case: dict, gen_sql: str,
               unlocked: dict[tuple[str, str], set[str]]) -> dict:
    """Score one reply: guard + result equality; gold errors never pass.

    Status notes carry no result rows: this repo is public and snapshot data
    must not reach stdout or reports.
    """
    gold_sql = case["sql"]
    refusal = case.get("group") == "refuse" or gold_sql.strip() == REFUSAL_TOKEN
    if refusal:
        passed = REFUSAL_TOKEN in gen_sql.upper()
        return {"passed": passed, "gold_error": False,
                "status": "PASS" if passed else "EXPECTED_REFUSAL"}
    try:
        gold_rows = conn.execute(gold_sql).fetchall()
    except Exception as exc:
        return {"passed": False, "gold_error": True, "status": f"GOLD_ERROR ({exc})"}

    guard_ok, checked_or_reason = check_sql_guard(gen_sql, unlocked)
    if not guard_ok:
        return {"passed": False, "gold_error": False,
                "status": f"GUARD_BLOCKED ({checked_or_reason})"}
    try:
        actual_rows = conn.execute(checked_or_reason).fetchall()
    except Exception as exc:
        return {"passed": False, "gold_error": False, "status": f"SQL_ERROR ({exc})"}

    if not actual_rows and not gold_rows:
        # Two empty results pass only when the SQL itself is equivalent.
        passed = sql_equivalent(gold_sql, checked_or_reason)
        status = "PASS" if passed else "EMPTY_MISMATCH"
    else:
        passed = results_equal(actual_rows, gold_rows, gold_is_ordered(gold_sql))
        status = "PASS" if passed else "ROW_MISMATCH"
    return {"passed": passed, "gold_error": False, "status": status}


def case_groups(cases: list[dict]) -> list[str]:
    """Group names in first-seen order, for dynamic report columns."""
    return list(dict.fromkeys(case.get("group") or "other" for case in cases))


def evaluate_condition(conn: sqlite3.Connection, cases: list[dict], mode_label: str,
                       retrieve, generate=None) -> dict:
    """Score one condition; ``retrieve(question)`` returns RAG chunks and
    ``generate(question, chunks)`` returns the model SQL (or CANNOT_ANSWER)."""
    if generate is None:
        generate = ollama_generate_sql
    groups = case_groups(cases)
    group_counts = {g: [0, 0] for g in groups}
    passed_count = 0
    gold_errors = 0
    case_records = []
    t_mode = time.time()
    print(f"\n--- Condition: {mode_label} ---")
    for case in cases:
        t0 = time.time()
        q = case["question"]
        chunks = retrieve(q)
        unlocked = extract_unlocked_codes(chunks)
        gen_sql = generate(q, chunks)
        scored = score_case(conn, case, gen_sql, unlocked)
        passed = scored["passed"]
        if passed:
            passed_count += 1
            group_counts[case.get("group") or "other"][0] += 1
        group_counts[case.get("group") or "other"][1] += 1
        if scored["gold_error"]:
            gold_errors += 1
        elapsed = round(time.time() - t0, 2)
        print(f"   [{case['id']}] {'PASS' if passed else 'FAIL'} ({elapsed}s) | "
              f"Q: {q} -> {gen_sql} | {scored['status']}")
        case_records.append(
            {
                "id": case["id"],
                "question": q,
                "gold_sql": case["sql"],
                "generated_sql": gen_sql,
                "passed": passed,
                "gold_error": scored["gold_error"],
                "status": scored["status"],
                "elapsed_s": elapsed,
            }
        )

    total_time = round(time.time() - t_mode, 2)
    rate = 100.0 * passed_count / len(cases) if cases else 0.0
    print(f"   >>> Score for {mode_label}: {passed_count}/{len(cases)} ({rate:.1f}%) "
          f"[gold_error: {gold_errors}] in {total_time}s")
    return {
        "label": mode_label,
        "passed": passed_count,
        "total": len(cases),
        "gold_error": gold_errors,
        "pass_rate_pct": round(rate, 1),
        "seconds": total_time,
        "groups": group_counts,
        "cases": case_records,
    }


def build_eval_report_lines(cases: list[dict], eval_summary: dict, db_note: str) -> list[str]:
    """Section 2 of the report: dynamic case count, group columns, per-case table."""
    groups = case_groups(cases)
    mode_keys = ("no_rag", "single_shard_node1", "decentralized_50_50_rag")
    lines = [
        "",
        f"## 2. Held-Out Evaluation (`eval/softone_holdout.jsonl`, {len(cases)} Cases, EN + EL, `qwen3.5:4b`)",
        "",
        db_note,
        "",
        "| Experimental Condition | Passed | Pass Rate | "
        + " | ".join(groups) + " | Gold Errors | Time |",
        "|" + "---|" * (5 + len(groups)),
    ]
    for mode_key in mode_keys:
        info = eval_summary[mode_key]
        cells = " | ".join(f"{info['groups'][g][0]}/{info['groups'][g][1]}" for g in groups)
        lines.append(
            f"| **{info['label']}** | **{info['passed']}/{info['total']}** | **{info['pass_rate_pct']}%** | "
            f"{cells} | {info['gold_error']} | {info['seconds']} s |"
        )

    lines += [
        "",
        "### Per-Case Breakdown across Topologies",
        "",
        "| Case ID | Question | No-RAG | Single-Shard (50%) | Decentralized 50/50 RAG | Generated SQL (Decentralized) |",
        "|---|---|---|---|---|---|",
    ]

    def cell(mode_key: str, idx: int) -> str:
        rec = eval_summary[mode_key]["cases"][idx]
        if rec["gold_error"]:
            return "GOLD_ERROR"
        return "PASS" if rec["passed"] else "FAIL"

    for idx, case in enumerate(cases):
        c_dec_rec = eval_summary["decentralized_50_50_rag"]["cases"][idx]
        lines.append(
            f"| `{case['id']}` | {case['question']} | {cell('no_rag', idx)} | "
            f"{cell('single_shard_node1', idx)} | **{cell('decentralized_50_50_rag', idx)}** | "
            f"`{c_dec_rec['generated_sql']}` |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-case end-to-end evaluation on the live cluster.")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT),
                        help="SoftOne snapshot scored read-only; falls back to the toy DB when missing")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N holdout cases")
    parser.add_argument("--out-dir", default=str(_ROOT / "results"),
                        help="directory for real_eval_raw.json and the Excel/CSV exports")
    args = parser.parse_args()

    certs = Path(_ROOT / "certs")
    ssl_ctx = make_client_ssl_context(
        str(certs / "ca.crt"),
        str(certs / "node1-maestro.crt"),
        str(certs / "node1-maestro.key"),
    )
    node_ids = ["node1-maestro", "node2-peer", "node3-peer"]
    rag_urls = {
        "node1-maestro": "https://localhost:18443",
        "node2-peer": "https://localhost:28443",
        "node3-peer": "https://localhost:38443",
    }
    raft_urls = {
        "node1-maestro": "https://localhost:17050",
        "node2-peer": "https://localhost:27050",
        "node3-peer": "https://localhost:37050",
    }
    node1_rag = rag_urls["node1-maestro"]
    ring = ShardRing(node_ids)

    print("=== 1. Connecting to Live 6-Container Cluster (Raft mTLS + Private Kubo IPFS + Ollama) ===")
    wait_for_nodes(
        ssl_ctx,
        [f"{url}/health" for url in raft_urls.values()] + [f"{url}/status" for url in rag_urls.values()],
    )
    connect_kubo_private_swarm("http://localhost:15001", "http://localhost:25001", "http://localhost:35001")

    facts = [
        json.loads(line)
        for line in (_ROOT / "eval/taught_facts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_seed = [
        json.loads(line)
        for line in (_ROOT / "eval/softone_train_seed.jsonl").read_text().splitlines()
        if line.strip()
    ]
    holdout = [
        json.loads(line)
        for line in (_ROOT / "eval/softone_holdout.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        holdout = holdout[: args.limit]

    # Four documents, one per ERP area; the hash ring decides which node owns each of them.
    trdr_facts = [f["text"] for f in facts if f["table_name"] == "TRDR"]
    mtrl_facts = [f["text"] for f in facts if f["table_name"] == "MTRL"]
    findoc_sosource_facts = [
        f["text"] for f in facts if f["table_name"] == "FINDOC" and f["column_name"] in ("SOSOURCE", "ISCANCEL")
    ]
    findoc_sodtype_facts = [
        f["text"] for f in facts if f["table_name"] == "FINDOC" and f["column_name"] == "SODTYPE"
    ]

    shard1_train = [{"prompt": r["question"], "target": r["sql"]} for r in train_seed[:8]]
    shard2_train = [{"prompt": r["question"], "target": r["sql"]} for r in train_seed[8:]]

    docs = [
        {
            "doc_id": "real-pdf-trdr",
            "title": "SoftOne ERP Trader Master (TRDR) Verified Code Specification",
            "page_count": 2,
            "chunks": trdr_facts,
            "training_pairs": shard1_train[:4],
        },
        {
            "doc_id": "real-pdf-mtrl",
            "title": "SoftOne ERP Material & Service Master (MTRL) Verified Code Specification",
            "page_count": 2,
            "chunks": mtrl_facts,
            "training_pairs": shard1_train[4:],
        },
        {
            "doc_id": "real-pdf-findoc-series",
            "title": "SoftOne ERP Financial Documents (FINDOC SOSOURCE & ISCANCEL) Verified Specification",
            "page_count": 3,
            "chunks": findoc_sosource_facts,
            "training_pairs": shard2_train[:4],
        },
        {
            "doc_id": "real-pdf-findoc-sides",
            "title": "SoftOne ERP Financial Document Sides (FINDOC SODTYPE) Verified Specification",
            "page_count": 2,
            "chunks": findoc_sodtype_facts,
            "training_pairs": shard2_train[4:],
        },
    ]

    print("\n=== 2. Ingesting Real SoftOne Domain PDFs (hash shards) into Kubo IPFS + Ollama Embeddings ===")
    ingested_manifests = []
    for doc in docs:
        assignment = ring.assignment(doc["doc_id"])
        pdf_bytes = make_minimal_pdf(
            doc["title"],
            doc["chunks"],
            doc_id=doc["doc_id"],
            shard_owner=assignment["owner"],
            metadata_holder=assignment["backup"],
        )
        res = mtls_request(
            ssl_ctx,
            f"{rag_urls[assignment['owner']]}/shard/ingest-full",
            method="POST",
            payload={**doc, "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii")},
        )
        ingested_manifests.append(res["manifest"])
        print(
            f"   [{assignment['shard_id']} -> {assignment['owner']}] {doc['doc_id']} | "
            f"CID={res['manifest']['ipfs_cid']} | chunks={len(doc['chunks'])} | "
            f"Raft block #{res['raft_block_index']}"
        )

    print(f"\n=== 3. Evaluating Real Held-Out Set ({len(holdout)} Cases, EN+EL) with qwen3.5:4b & embeddinggemma ===")
    conn, is_snapshot = open_eval_db(Path(args.snapshot))
    if is_snapshot:
        db_note = f"Scoring database: read-only SoftOne snapshot (`{args.snapshot}`)."
        print(f"Scoring on read-only snapshot: {args.snapshot}")
    else:
        db_note = ("**WARNING: toy database, numbers not comparable** — snapshot not found at "
                   f"`{args.snapshot}`; scored on the built-in `build_eval_sqlite()` rows.")
        print(f"WARNING: snapshot not found at {args.snapshot}; falling back to build_eval_sqlite() "
              "— toy database, numbers not comparable")
    # Mode keys are kept for results/real_eval_raw.json and the Excel exporter.
    modes = [
        ("no_rag", "0% PDFs (No RAG / Base qwen3.5:4b)"),
        ("single_shard_node1", "One shard only (node1-maestro's hash shard)"),
        ("decentralized_50_50_rag", "All shards (decentralized scatter-gather mTLS + IPFS)"),
    ]
    eval_summary: dict[str, dict] = {}

    def make_retrieve(mode_key: str):
        def retrieve(q: str) -> list[dict]:
            if mode_key == "no_rag":
                return []
            if mode_key == "single_shard_node1":
                r_local = mtls_request(
                    ssl_ctx, f"{node1_rag}/rag/search-local", method="POST", payload={"query": q, "k": 15}
                )
                return [{**c, "ledger_hash_verified": True} for c in r_local["results"]]
            r_dec = mtls_request(
                ssl_ctx,
                f"{node1_rag}/rag/query-decentralized",
                method="POST",
                payload={"query": q, "k": 15},
            )
            return r_dec["top_k"]
        return retrieve

    for mode_key, mode_label in modes:
        eval_summary[mode_key] = evaluate_condition(
            conn, holdout, mode_label, make_retrieve(mode_key)
        )

    print("\n=== 4. Running 5-Round Federated LoRA Ablation (FedAvg-LoRA vs. FFA-LoRA) on 16 Seed Pairs ===")
    lora_ablation: dict[str, list[dict]] = {"fedavg": [], "ffa": []}
    for lora_mode in ("fedavg", "ffa"):
        for r in range(1, 6):
            res = mtls_request(
                ssl_ctx,
                f"{node1_rag}/fed/run-round",
                method="POST",
                payload={"round_id": r, "lora_mode": lora_mode},
            )
            lora_ablation[lora_mode].append(res)
            per_client = " | ".join(
                f"{c['node_id']}: {c['first_loss']:.4f}->{c['last_loss']:.4f} ({c['num_examples']} ex)"
                for c in res["clients"]
            )
            print(
                f"   [{lora_mode.upper()} Round {r}] {per_client} | "
                f"Global Loss={res['weighted_loss']:.4f} | "
                f"participants={sorted(res['participants'])} | IPFS CID={res['global_adapter_cid']}"
            )

    statuses = {node_id: mtls_request(ssl_ctx, f"{rag_urls[node_id]}/status") for node_id in node_ids}
    s1, s2, s3 = statuses["node1-maestro"], statuses["node2-peer"], statuses["node3-peer"]

    # Write full markdown report to docs/REAL_EVAL_RESULTS.md
    report_lines = [
        "# Real-Case Experimental Results (MacBook Air M4 — 4-Container mTLS + Kubo IPFS + Ollama)",
        "",
        "Measured end-to-end on the live Docker cluster "
        "(`node1-maestro`, `node2-peer`, `node3-peer`, `ipfs-node1..3`, `ollama-node1..3`)",
        "connected to host Ollama (`embeddinggemma:latest` 768-d semantic embeddings + `qwen3.5:4b` SQL generator).",
        "",
        db_note,
        "",
        "## 1. Hash-Sharded PDF Corpus & Private Kubo IPFS CIDs (`LIBP2P_FORCE_PNET=1`)",
        "",
        "| Document ID | Shard Owner | Metadata + Backup Peer | Kubo IPFS CIDv1 | PDF SHA-256 | Chunks |",
        "|---|---|---|---|---|---|",
    ]
    for m in ingested_manifests:
        report_lines.append(
            f"| `{m['doc_id']}` | `{m['shard_owner']}` | `{m['metadata_holder']}` | "
            f"`{m['ipfs_cid']}` | `{m['pdf_sha256'][:16]}...` | {m['num_chunks']} |"
        )

    report_lines += build_eval_report_lines(holdout, eval_summary, db_note)

    report_lines += [
        "",
        "## 3. Federated LoRA Training Ablation (`FedAvg-LoRA` vs. `FFA-LoRA`, seed pairs per hash shard)",
        "",
        "| Round | `FedAvg-LoRA` Loss | `FedAvg-LoRA` Global CID | `FFA-LoRA` Loss | `FFA-LoRA` Global CID |",
        "|---|---|---|---|---|",
    ]
    for r_idx in range(5):
        fa = lora_ablation["fedavg"][r_idx]
        ff = lora_ablation["ffa"][r_idx]
        report_lines.append(
            f"| {r_idx + 1} | `{fa['weighted_loss']:.4f}` | `{fa['global_adapter_cid']}` | "
            f"`{ff['weighted_loss']:.4f}` | `{ff['global_adapter_cid']}` |"
        )

    report_lines += ["", "## 4. Final Cluster & Ledger State", ""]
    for node_id, status in statuses.items():
        report_lines.append(
            f"- **`{node_id}`:** shard `{status['rag_store']['shard']['shard_id']}`, "
            f"Raft Height = `{status['raft']['ledger']['height']}`, "
            f"Tip Hash = `{status['raft']['ledger']['last_hash'][:20]}...`, "
            f"Kubo Pins = `{len(status['ipfs']['pins'])}`"
        )

    report_path = _ROOT / "docs/REAL_EVAL_RESULTS.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nWrote real evaluation report to {report_path}")

    results_dir = Path(args.out_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = results_dir / "real_eval_raw.json"
    raw_json_path.write_text(
        json.dumps(
            {
                "ingested_manifests": ingested_manifests,
                "eval_summary": eval_summary,
                "lora_ablation": lora_ablation,
                "node1_status": s1,
                "node2_status": s2,
                "node3_status": s3,
            },
            indent=2,
        )
    )
    from scripts.export_excel_results import export_all_from_json

    xlsx_path = export_all_from_json(raw_json_path, results_dir)
    print(f"Wrote Excel workbook and CSVs to {xlsx_path}")


if __name__ == "__main__":
    main()
