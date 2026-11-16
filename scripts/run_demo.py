#!/usr/bin/env python3
"""Modular CLI & End-to-End Demo Runner for the 2-Docker-Node mTLS Raft + Sharded RAG + Federated LoRA cluster."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fedrag.node_service import make_client_ssl_context, mtls_request  # noqa: E402


def make_minimal_pdf(title: str, body_lines: list[str]) -> bytes:
    """Create a valid, deterministic PDF-1.4 binary containing the document title and text."""
    content_text = "\\n".join([title, *body_lines])
    stream = f"BT /F1 12 Tf 72 720 Td ({content_text}) Tj ET".encode()
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
        + f"4 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
        + stream
        + b"\nendstream endobj\n"
        b"trailer << /Root 1 0 R >>\n%%EOF\n"
    )
    return pdf


CORPUS_SHARD_1 = [
    {
        "doc_id": "doc-101",
        "title": "SoftOne Customer & Supplier Type Codes (Shard 1)",
        "page_count": 3,
        "chunks": [
            "In the TRDR table, SODTYPE = 13 identifies customer records across all fiscal periods.",
            "In the TRDR table, SODTYPE = 12 identifies supplier and vendor records.",
        ],
        "training_pairs": [
            {
                "prompt": "How many customers exist in the TRDR table?",
                "target": "SELECT COUNT(*) FROM TRDR WHERE SODTYPE = 13;",
            },
            {
                "prompt": "List all active suppliers from TRDR.",
                "target": "SELECT CODE, NAME FROM TRDR WHERE SODTYPE = 12 AND ISACTIVE = 1 LIMIT 50;",
            },
        ],
    },
    {
        "doc_id": "doc-102",
        "title": "Sales Invoice & Cancellation Rules (Shard 1)",
        "page_count": 4,
        "chunks": [
            "Sales documents in FINDOC use SOSOURCE = 1351 and exclude cancelled rows with ISCANCEL = 0.",
            "Gross sales revenue is computed as SUM(NETAMNT) from FINDOC where SOSOURCE = 1351.",
        ],
        "training_pairs": [
            {
                "prompt": "What is the total net sales amount excluding cancellations?",
                "target": "SELECT SUM(NETAMNT) FROM FINDOC WHERE SOSOURCE = 1351 AND ISCANCEL = 0;",
            },
            {
                "prompt": "Count non-cancelled sales invoices in FINDOC.",
                "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1351 AND ISCANCEL = 0;",
            },
        ],
    },
    {
        "doc_id": "doc-103",
        "title": "Inventory & Material Master Specification (Shard 1)",
        "page_count": 2,
        "chunks": [
            "Inventory goods in MTRL have SODTYPE = 51 while services use SODTYPE = 52.",
            "Active inventory items are filtered with MTRL.ISACTIVE = 1.",
        ],
        "training_pairs": [
            {
                "prompt": "How many inventory goods are stored in MTRL?",
                "target": "SELECT COUNT(*) FROM MTRL WHERE SODTYPE = 51;",
            },
            {
                "prompt": "How many service items are stored in MTRL?",
                "target": "SELECT COUNT(*) FROM MTRL WHERE SODTYPE = 52;",
            },
        ],
    },
]

CORPUS_SHARD_2 = [
    {
        "doc_id": "doc-201",
        "title": "Purchase Orders & Supplier Ledger (Shard 2)",
        "page_count": 5,
        "chunks": [
            "Purchase invoices in FINDOC are recorded with SOSOURCE = 1251 and linked to TRDR via TRDR = TRDR.TRDR.",
            "Cancelled supplier invoices carry ISCANCEL = 1 in FINDOC.",
        ],
        "training_pairs": [
            {
                "prompt": "Count valid purchase invoices in FINDOC.",
                "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1251 AND ISCANCEL = 0;",
            },
            {
                "prompt": "Sum total purchase net amount from suppliers.",
                "target": "SELECT SUM(NETAMNT) FROM FINDOC WHERE SOSOURCE = 1251 AND ISCANCEL = 0;",
            },
        ],
    },
    {
        "doc_id": "doc-202",
        "title": "AADE myDATA e-Invoicing Transmission Series (Shard 2)",
        "page_count": 3,
        "chunks": [
            "Transmitted AADE myDATA invoice series in FINDOC use SOSOURCE = 1361 for certified B2B issuance.",
            "Draft invoices are prepared locally and issued exclusively through certified providers.",
        ],
        "training_pairs": [
            {
                "prompt": "How many transmitted series 1361 documents exist in FINDOC?",
                "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1361 AND ISCANCEL = 0;",
            },
            {
                "prompt": "Show latest 10 transmitted invoices in FINDOC.",
                "target": (
                    "SELECT FINDOC, TRNDATE, NETAMNT FROM FINDOC "
                    "WHERE SOSOURCE = 1361 ORDER BY TRNDATE DESC LIMIT 10;"
                ),
            },
        ],
    },
    {
        "doc_id": "doc-203",
        "title": "Multi-Branch Payment & Cash Collections (Shard 2)",
        "page_count": 2,
        "chunks": [
            "Customer cash collection receipts in FINDOC use SOSOURCE = 1381 across company branches.",
            "Each receipt references the customer account via FINDOC.TRDR.",
        ],
        "training_pairs": [
            {
                "prompt": "Count customer collection receipts with code 1381.",
                "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1381 AND ISCANCEL = 0;",
            },
            {
                "prompt": "What is the total collected amount for SOSOURCE 1381?",
                "target": "SELECT SUM(NETAMNT) FROM FINDOC WHERE SOSOURCE = 1381 AND ISCANCEL = 0;",
            },
        ],
    },
]


def wait_for_nodes(ssl_ctx, urls: list[str], timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    for url in urls:
        while True:
            try:
                mtls_request(ssl_ctx, url, method="GET", timeout=3.0)
                break
            except Exception as exc:
                if time.time() >= deadline:
                    raise RuntimeError(f"Timed out waiting for {url}: {exc}") from exc
                time.sleep(0.5)


def action_health(ssl_ctx, args) -> None:
    wait_for_nodes(
        ssl_ctx,
        [
            f"{args.node1_raft}/health",
            f"{args.node2_raft}/health",
            f"{args.node1_rag}/status",
            f"{args.node2_rag}/status",
        ],
    )
    h1 = mtls_request(ssl_ctx, f"{args.node1_raft}/health")
    h2 = mtls_request(ssl_ctx, f"{args.node2_raft}/health")
    print(f"   - node1-maestro Raft role={h1['role']}, client_cn={h1['verified_client_cn']}")
    print(f"   - node2-peer    Raft role={h2['role']}, client_cn={h2['verified_client_cn']}")


def action_ingest(ssl_ctx, args) -> None:
    print("Ingesting 50/50 Sharded PDFs + Embeddings + Cross-Node Backups over mTLS...")
    for doc in CORPUS_SHARD_1:
        pdf_bytes = make_minimal_pdf(doc["title"], doc["chunks"])
        res = mtls_request(
            ssl_ctx,
            f"{args.node1_rag}/shard/ingest-full",
            method="POST",
            payload={
                **doc,
                "metadata_holder": "node2-peer",
                "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            },
        )
        print(
            f"   [Shard 1 -> node1-maestro] {doc['doc_id']} full PDF+vectors stored; "
            f"metadata stub + backup verified on node2-peer (Raft block #{res['raft_block_index']}, "
            f"sha256={res['manifest']['pdf_sha256'][:12]}...)"
        )

    for doc in CORPUS_SHARD_2:
        pdf_bytes = make_minimal_pdf(doc["title"], doc["chunks"])
        res = mtls_request(
            ssl_ctx,
            f"{args.node2_rag}/shard/ingest-full",
            method="POST",
            payload={
                **doc,
                "metadata_holder": "node1-maestro",
                "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            },
        )
        print(
            f"   [Shard 2 -> node2-peer]    {doc['doc_id']} full PDF+vectors stored; "
            f"metadata stub + backup verified on node1-maestro (Raft block #{res['raft_block_index']}, "
            f"sha256={res['manifest']['pdf_sha256'][:12]}...)"
        )


def action_train(ssl_ctx, args) -> None:
    print(
        f"Running {args.rounds} Federated LoRA Training Rounds "
        f"(mode={args.lora_mode}) across Node 1 (Shard 1) & Node 2 (Shard 2)..."
    )
    for r in range(1, args.rounds + 1):
        round_res = mtls_request(
            ssl_ctx,
            f"{args.node1_rag}/fed/run-round",
            method="POST",
            payload={"round_id": r, "lora_mode": args.lora_mode},
        )
        c1, c2 = round_res["clients"][0], round_res["clients"][1]
        print(
            f"   [Round {r} | {args.lora_mode}] "
            f"{c1['node_id']} loss {c1['first_loss']:.4f} -> {c1['last_loss']:.4f} ({c1['num_examples']} ex) | "
            f"{c2['node_id']} loss {c2['first_loss']:.4f} -> {c2['last_loss']:.4f} ({c2['num_examples']} ex) | "
            f"FedAvg weighted_loss={round_res['weighted_loss']:.4f} | "
            f"global_sha256={round_res['global_adapter_sha256'][:12]}... (Raft block #{round_res['raft_block_index']})"
        )


def action_query(ssl_ctx, args) -> None:
    queries = (
        [args.query]
        if args.query
        else [
            "Which SODTYPE code identifies customer records in TRDR?",
            "What SOSOURCE code is used for purchase invoices from suppliers in FINDOC?",
        ]
    )
    print("Running Decentralized Scatter-Gather RAG Queries across Both Shards...")
    for q in queries:
        q_res = mtls_request(
            ssl_ctx,
            f"{args.node1_rag}/rag/query-decentralized",
            method="POST",
            payload={"query": q, "k": 2},
        )
        top = q_res["top_k"][0]
        print(
            f"   Query: {q!r}\n"
            f"     -> Top Hit from {top['node_id']} ({top['doc_id']}): {top['text']!r}\n"
            f"     -> Cosine Score={top['score']} | On-Chain SHA-256 Verified={top['ledger_hash_verified']}"
        )


def action_tamper(ssl_ctx, args) -> None:
    print("Verifying Tamper Rejection on Raft Ledger...")
    tampered_rejected = False
    try:
        mtls_request(
            ssl_ctx,
            f"{args.node1_raft}/tx/propose",
            method="POST",
            payload={
                "tx_type": "BACKUP_CHECK",
                "payload": {
                    "doc_id": "doc-101",
                    "source_node": "node1-maestro",
                    "backup_node": "node2-peer",
                    "computed_sha256": "f" * 64,
                },
            },
        )
    except RuntimeError as exc:
        tampered_rejected = True
        print(f"   -> Tampered backup checksum properly rejected by Raft chaincode: {exc}")

    if not tampered_rejected:
        raise RuntimeError("Expected tampered backup checksum to be rejected!")


def action_status(ssl_ctx, args) -> None:
    s1 = mtls_request(ssl_ctx, f"{args.node1_rag}/status")
    s2 = mtls_request(ssl_ctx, f"{args.node2_rag}/status")
    h1_end = s1["raft"]["ledger"]["height"]
    h2_end = s2["raft"]["ledger"]["height"]
    hash1 = s1["raft"]["ledger"]["last_hash"]
    hash2 = s2["raft"]["ledger"]["last_hash"]

    print("Cluster Consistency & Storage Check:")
    print(
        f"   - node1-maestro: full_docs={len(s1['rag_store']['full_shard_documents'])}, "
        f"metadata_stubs={len(s1['rag_store']['metadata_only_stubs'])}, "
        f"verified_backups={len(s1['rag_store']['verified_backups'])}, "
        f"raft_height={h1_end}, tip_hash={hash1[:16]}..."
    )
    print(
        f"   - node2-peer:    full_docs={len(s2['rag_store']['full_shard_documents'])}, "
        f"metadata_stubs={len(s2['rag_store']['metadata_only_stubs'])}, "
        f"verified_backups={len(s2['rag_store']['verified_backups'])}, "
        f"raft_height={h2_end}, tip_hash={hash2[:16]}..."
    )
    assert h1_end == h2_end and hash1 == hash2, "Raft ledger state diverged between nodes!"


def action_ledger(ssl_ctx, args) -> None:
    snap = mtls_request(ssl_ctx, f"{args.node1_raft}/ledger")
    print(json.dumps(snap, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2-Docker-node mTLS Federated LoRA & Sharded RAG CLI")
    parser.add_argument(
        "--action",
        default="all",
        choices=("all", "health", "ingest", "train", "query", "tamper", "status", "ledger"),
    )
    parser.add_argument("--node1-rag", default="https://localhost:18443")
    parser.add_argument("--node2-rag", default="https://localhost:28443")
    parser.add_argument("--node1-raft", default="https://localhost:17050")
    parser.add_argument("--node2-raft", default="https://localhost:27050")
    parser.add_argument("--certs-dir", default="./certs")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lora-mode", default="fedavg", choices=("fedavg", "ffa"))
    parser.add_argument("--query", default="")
    args = parser.parse_args()

    certs = Path(args.certs_dir)
    ssl_ctx = make_client_ssl_context(
        str(certs / "ca.crt"),
        str(certs / "node1-maestro.crt"),
        str(certs / "node1-maestro.key"),
    )

    if args.action == "health":
        action_health(ssl_ctx, args)
    elif args.action == "ingest":
        action_ingest(ssl_ctx, args)
    elif args.action == "train":
        action_train(ssl_ctx, args)
    elif args.action == "query":
        action_query(ssl_ctx, args)
    elif args.action == "tamper":
        action_tamper(ssl_ctx, args)
    elif args.action == "status":
        action_status(ssl_ctx, args)
    elif args.action == "ledger":
        action_ledger(ssl_ctx, args)
    elif args.action == "all":
        print("1. Waiting for mTLS Raft & Python RAG nodes...")
        action_health(ssl_ctx, args)
        print()
        action_ingest(ssl_ctx, args)
        print()
        action_train(ssl_ctx, args)
        print()
        action_query(ssl_ctx, args)
        print()
        action_tamper(ssl_ctx, args)
        print()
        action_status(ssl_ctx, args)
        print("\nSUCCESS: 2-Docker mTLS Raft + 50/50 Sharded RAG + Federated LoRA test passed!")


if __name__ == "__main__":
    main()
