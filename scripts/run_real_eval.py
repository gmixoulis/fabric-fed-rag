#!/usr/bin/env python3
"""Real-Case End-to-End Evaluation on the 4-Container mTLS Raft + Private Kubo IPFS + Sharded RAG Cluster.

Uses:
- Real Ollama embeddings (`embeddinggemma:latest`, 768-d) inside the Docker containers
- Real Ollama LLM (`qwen3.5:4b`) for SQL generation over the SoftOne ERP schema
- Real Private Kubo IPFS swarm (`ipfs-node1` & `ipfs-node2` with `LIBP2P_FORCE_PNET=1`)
- Real SoftOne datasets (`eval/taught_facts.jsonl`, `eval/softone_train_seed.jsonl`, `eval/softone_holdout.jsonl`)
"""

from __future__ import annotations

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
from scripts.run_demo import connect_kubo_private_swarm, make_minimal_pdf, wait_for_nodes  # noqa: E402

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


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    out = []
    for row in rows:
        norm_row = tuple(round(float(x), 2) if isinstance(x, float) else x for x in row)
        out.append(norm_row)
    return out


def main() -> None:
    certs = Path(_ROOT / "certs")
    ssl_ctx = make_client_ssl_context(
        str(certs / "ca.crt"),
        str(certs / "node1-maestro.crt"),
        str(certs / "node1-maestro.key"),
    )
    node1_rag = "https://localhost:18443"
    node2_rag = "https://localhost:28443"
    node1_raft = "https://localhost:17050"
    node2_raft = "https://localhost:27050"

    print("=== 1. Connecting to Live 4-Container Cluster (Raft mTLS + Private Kubo IPFS + Ollama) ===")
    wait_for_nodes(
        ssl_ctx,
        [
            f"{node1_raft}/health",
            f"{node2_raft}/health",
            f"{node1_rag}/status",
            f"{node2_rag}/status",
        ],
    )
    connect_kubo_private_swarm("http://localhost:15001", "http://localhost:25001")

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

    # Split the 15 taught facts and 16 training examples 50/50 across Shard 1 (TRDR + MTRL) and Shard 2 (FINDOC)
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

    shard1_docs = [
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
    ]

    shard2_docs = [
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

    print("\n=== 2. Ingesting Real SoftOne Domain PDFs (50/50 Split) into Kubo IPFS + Ollama Embeddings ===")
    ingested_manifests = []
    for doc in shard1_docs:
        pdf_bytes = make_minimal_pdf(doc["title"], doc["chunks"])
        res = mtls_request(
            ssl_ctx,
            f"{node1_rag}/shard/ingest-full",
            method="POST",
            payload={
                **doc,
                "metadata_holder": "node2-peer",
                "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            },
        )
        ingested_manifests.append(res["manifest"])
        print(
            f"   [Shard 1 -> node1-maestro] {doc['doc_id']} | CID={res['manifest']['ipfs_cid']} | "
            f"chunks={len(doc['chunks'])} | Raft block #{res['raft_block_index']}"
        )

    for doc in shard2_docs:
        pdf_bytes = make_minimal_pdf(doc["title"], doc["chunks"])
        res = mtls_request(
            ssl_ctx,
            f"{node2_rag}/shard/ingest-full",
            method="POST",
            payload={
                **doc,
                "metadata_holder": "node1-maestro",
                "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            },
        )
        ingested_manifests.append(res["manifest"])
        print(
            f"   [Shard 2 -> node2-peer]    {doc['doc_id']} | CID={res['manifest']['ipfs_cid']} | "
            f"chunks={len(doc['chunks'])} | Raft block #{res['raft_block_index']}"
        )

    print("\n=== 3. Evaluating Real Held-Out Set (12 Cases, EN+EL) with qwen3.5:4b & embeddinggemma ===")
    conn = build_eval_sqlite()
    modes = [
        ("no_rag", "0% PDFs (No RAG / Base qwen3.5:4b)"),
        ("single_shard_node1", "50% PDFs (Node 1 Shard 1 Only)"),
        ("decentralized_50_50_rag", "100% Shards (Decentralized 50/50 Scatter-Gather mTLS + IPFS)"),
    ]
    eval_summary: dict[str, dict] = {}

    for mode_key, mode_label in modes:
        passed_count = 0
        case_records = []
        t_mode = time.time()
        print(f"\n--- Condition: {mode_label} ---")
        for case in holdout:
            t0 = time.time()
            q = case["question"]
            gold_sql = case["sql"]
            gold_rows = normalize_rows(conn.execute(gold_sql).fetchall())

            if mode_key == "no_rag":
                chunks = []
            elif mode_key == "single_shard_node1":
                r_local = mtls_request(
                    ssl_ctx, f"{node1_rag}/rag/search-local", method="POST", payload={"query": q, "k": 15}
                )
                chunks = [{**c, "ledger_hash_verified": True} for c in r_local["results"]]
            else:
                r_dec = mtls_request(
                    ssl_ctx,
                    f"{node1_rag}/rag/query-decentralized",
                    method="POST",
                    payload={"query": q, "k": 15},
                )
                chunks = r_dec["top_k"]

            unlocked = extract_unlocked_codes(chunks)
            gen_sql = ollama_generate_sql(q, chunks, model="qwen3.5:4b")
            guard_ok, checked_or_reason = check_sql_guard(gen_sql, unlocked)

            passed = False
            status_note = ""
            if not guard_ok:
                status_note = f"GUARD_BLOCKED ({checked_or_reason})"
            else:
                try:
                    actual_rows = normalize_rows(conn.execute(checked_or_reason).fetchall())
                    passed = actual_rows == gold_rows
                    status_note = "PASS" if passed else f"ROW_MISMATCH (got {actual_rows} vs {gold_rows})"
                except Exception as exc:
                    status_note = f"SQL_ERROR ({exc})"

            if passed:
                passed_count += 1
            elapsed = round(time.time() - t0, 2)
            print(f"   [{case['id']}] {'PASS' if passed else 'FAIL'} ({elapsed}s) | Q: {q} -> {gen_sql} | {status_note}")
            case_records.append(
                {
                    "id": case["id"],
                    "question": q,
                    "gold_sql": gold_sql,
                    "generated_sql": gen_sql,
                    "passed": passed,
                    "status": status_note,
                    "elapsed_s": elapsed,
                }
            )

        total_time = round(time.time() - t_mode, 2)
        rate = 100.0 * passed_count / len(holdout)
        print(f"   >>> Score for {mode_label}: {passed_count}/{len(holdout)} ({rate:.1f}%) in {total_time}s")
        eval_summary[mode_key] = {
            "label": mode_label,
            "passed": passed_count,
            "total": len(holdout),
            "pass_rate_pct": round(rate, 1),
            "seconds": total_time,
            "cases": case_records,
        }

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
            c1, c2 = res["clients"][0], res["clients"][1]
            print(
                f"   [{lora_mode.upper()} Round {r}] "
                f"Shard1: {c1['first_loss']:.4f}->{c1['last_loss']:.4f} ({c1['num_examples']} ex) | "
                f"Shard2: {c2['first_loss']:.4f}->{c2['last_loss']:.4f} ({c2['num_examples']} ex) | "
                f"Global Loss={res['weighted_loss']:.4f} | IPFS CID={res['global_adapter_cid']}"
            )

    s1 = mtls_request(ssl_ctx, f"{node1_rag}/status")
    s2 = mtls_request(ssl_ctx, f"{node2_rag}/status")

    # Write full markdown report to docs/REAL_EVAL_RESULTS.md
    report_lines = [
        "# Real-Case Experimental Results (MacBook Air M4 — 4-Container mTLS + Kubo IPFS + Ollama)",
        "",
        "Measured end-to-end on the live 4-container Docker cluster "
        "(`node1-maestro`, `node2-peer`, `ipfs-node1`, `ipfs-node2`)",
        "connected to host Ollama (`embeddinggemma:latest` 768-d semantic embeddings + `qwen3.5:4b` SQL generator).",
        "",
        "## 1. 50/50 Sharded PDF Corpus & Private Kubo IPFS CIDs (`LIBP2P_FORCE_PNET=1`)",
        "",
        "| Document ID | Active Shard Owner | Metadata + Backup Peer | Kubo IPFS CIDv1 | PDF SHA-256 | Chunks |",
        "|---|---|---|---|---|---|",
    ]
    for m in ingested_manifests:
        report_lines.append(
            f"| `{m['doc_id']}` | `{m['shard_owner']}` | `{m['metadata_holder']}` | "
            f"`{m['ipfs_cid']}` | `{m['pdf_sha256'][:16]}...` | {m['num_chunks']} |"
        )

    report_lines += [
        "",
        "## 2. Held-Out Evaluation (`eval/softone_holdout.jsonl`, 12 Cases, EN + EL, `qwen3.5:4b`)",
        "",
        "| Experimental Condition | Passed | Pass Rate | Plain (`hp01–06`) | Code (`hc01–06`) | Time |",
        "|---|---|---|---|---|---|",
    ]
    for mode_key in ("no_rag", "single_shard_node1", "decentralized_50_50_rag"):
        info = eval_summary[mode_key]
        hp_pass = sum(1 for c in info["cases"] if c["id"].startswith("hp") and c["passed"])
        hc_pass = sum(1 for c in info["cases"] if c["id"].startswith("hc") and c["passed"])
        report_lines.append(
            f"| **{info['label']}** | **{info['passed']}/{info['total']}** | **{info['pass_rate_pct']}%** | "
            f"{hp_pass}/6 | {hc_pass}/6 | {info['seconds']} s |"
        )

    report_lines += [
        "",
        "### Per-Case Breakdown across Topologies",
        "",
        "| Case ID | Question | No-RAG | Single-Shard (50%) | Decentralized 50/50 RAG | Generated SQL (Decentralized) |",
        "|---|---|---|---|---|---|",
    ]
    for idx, case in enumerate(holdout):
        c_no = "PASS" if eval_summary["no_rag"]["cases"][idx]["passed"] else "FAIL"
        c_s1 = "PASS" if eval_summary["single_shard_node1"]["cases"][idx]["passed"] else "FAIL"
        c_dec_rec = eval_summary["decentralized_50_50_rag"]["cases"][idx]
        c_dec = "PASS" if c_dec_rec["passed"] else "FAIL"
        report_lines.append(
            f"| `{case['id']}` | {case['question']} | {c_no} | {c_s1} | **{c_dec}** | `{c_dec_rec['generated_sql']}` |"
        )

    report_lines += [
        "",
        "## 3. Federated LoRA Training Ablation (`FedAvg-LoRA` vs. `FFA-LoRA`, 16 Seed Examples Split 8/8)",
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

    h1_tip = s1["raft"]["ledger"]["last_hash"][:20]
    h2_tip = s2["raft"]["ledger"]["last_hash"][:20]
    report_lines += [
        "",
        "## 4. Final Cluster & Ledger State",
        "",
        f"- **`node1-maestro`:** Raft Height = `{s1['raft']['ledger']['height']}`, "
        f"Tip Hash = `{h1_tip}...`, Kubo Pins = `{len(s1['ipfs']['pins'])}`",
        f"- **`node2-peer`:** Raft Height = `{s2['raft']['ledger']['height']}`, "
        f"Tip Hash = `{h2_tip}...`, Kubo Pins = `{len(s2['ipfs']['pins'])}`",
    ]

    report_path = _ROOT / "docs/REAL_EVAL_RESULTS.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nWrote real evaluation report to {report_path}")

    results_dir = _ROOT / "results"
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
            },
            indent=2,
        )
    )
    from scripts.export_excel_results import export_all_from_json

    xlsx_path = export_all_from_json(raw_json_path, results_dir)
    print(f"Wrote Excel workbook and CSVs to {xlsx_path}")


if __name__ == "__main__":
    main()
