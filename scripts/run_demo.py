#!/usr/bin/env python3
"""Modular CLI & End-to-End Demo Runner for the 3-node mTLS Raft + Sharded RAG + Federated LoRA cluster.

Documents are routed to their hash owner (`fedrag.shard_store.ShardRing`), so the same
corpus works unchanged with two or three nodes.
"""

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

from fedrag.ipfs_client import KuboIPFSClient  # noqa: E402
from fedrag.node_service import make_client_ssl_context, mtls_request  # noqa: E402
from fedrag.pdf_embedder import build_full_pdf_with_metadata  # noqa: E402
from fedrag.shard_store import ShardRing  # noqa: E402


def make_minimal_pdf(
    title: str,
    body_lines: list[str],
    *,
    doc_id: str = "doc-000",
    shard_owner: str = "node1-maestro",
    metadata_holder: str = "node2-peer",
    page_count: int = 1,
) -> bytes:
    """Create a valid PDF-1.4 binary with embedded `/Info` metadata hashes (`content_sha256`, `chunks_sha256`)."""
    pdf_bytes, _meta = build_full_pdf_with_metadata(
        doc_id=doc_id,
        title=title,
        shard_owner=shard_owner,
        metadata_holder=metadata_holder,
        chunks=body_lines,
        page_count=page_count,
    )
    return pdf_bytes


# Nine synthetic documents: three per shard at three nodes (4/5 at two nodes, see
# `docs/EXPERIMENTS.md` E9). No client data, no real snapshot rows.
CORPUS = [
    {
        "doc_id": "doc-121",
        "title": "SoftOne Trader Type Codes (TRDR)",
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
        "doc_id": "doc-122",
        "title": "Sales Invoice & Cancellation Rules (FINDOC)",
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
        "doc_id": "doc-123",
        "title": "Inventory & Material Master Specification (MTRL)",
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
    {
        "doc_id": "doc-124",
        "title": "Purchase Orders & Supplier Ledger (FINDOC)",
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
        "doc_id": "doc-125",
        "title": "AADE myDATA e-Invoicing Transmission Series",
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
        "doc_id": "doc-126",
        "title": "Multi-Branch Payment & Cash Collections",
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
    {
        "doc_id": "doc-127",
        "title": "Warehouse Stock Balances (MTRLINES)",
        "page_count": 3,
        "chunks": [
            "Warehouse stock valuation sums MTRLINES.LINEVAL grouped by FINDOC.BRANCH.",
            "Quantities per line are stored in MTRLINES.QTY1 and priced in MTRLINES.PRICE.",
        ],
        "training_pairs": [
            {
                "prompt": "What is the total stock value per branch?",
                "target": "SELECT BRANCH, SUM(LINEVAL) FROM MTRLINES GROUP BY BRANCH;",
            },
            {
                "prompt": "How many stock lines have quantity above ten?",
                "target": "SELECT COUNT(*) FROM MTRLINES WHERE QTY1 > 10;",
            },
        ],
    },
    {
        "doc_id": "doc-128",
        "title": "VAT Totals & Document Sums (FINDOC)",
        "page_count": 2,
        "chunks": [
            "Document VAT totals are stored in FINDOC.VATAMNT and gross totals in FINDOC.SUMAMNT.",
            "SUMAMNT equals NETAMNT plus VATAMNT for every non-cancelled document.",
        ],
        "training_pairs": [
            {
                "prompt": "What is the total VAT amount of all sales invoices?",
                "target": "SELECT SUM(VATAMNT) FROM FINDOC WHERE SOSOURCE = 1351 AND ISCANCEL = 0;",
            },
            {
                "prompt": "Which documents have a gross total above one thousand?",
                "target": "SELECT FINDOC FROM FINDOC WHERE SUMAMNT > 1000;",
            },
        ],
    },
    {
        "doc_id": "doc-129",
        "title": "Credit Notes & Returns (FINDOC SODTYPE)",
        "page_count": 2,
        "chunks": [
            "Credit notes carry FINDOC.SODTYPE = 5 and reference the original invoice in FINCODE.",
            "Returned goods are recorded as negative NETAMNT lines in MTRLINES.",
        ],
        "training_pairs": [
            {
                "prompt": "How many credit notes exist?",
                "target": "SELECT COUNT(*) FROM FINDOC WHERE SODTYPE = 5;",
            },
            {
                "prompt": "What is the total value of returned goods?",
                "target": "SELECT SUM(LINEVAL) FROM MTRLINES WHERE LINEVAL < 0;",
            },
        ],
    },
]


def shard_ring(args) -> ShardRing:
    return ShardRing([args.node1_id, args.node2_id, args.node3_id])


def rag_urls(args) -> dict[str, str]:
    return {
        args.node1_id: args.node1_rag,
        args.node2_id: args.node2_rag,
        args.node3_id: args.node3_rag,
    }


def wait_for_nodes(ssl_ctx, urls: list[str], timeout_s: float = 45.0) -> None:
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


def connect_kubo_private_swarm(ipfs1_url: str, ipfs2_url: str, ipfs3_url: str = "", timeout_s: float = 30.0) -> None:
    clients = [KuboIPFSClient(ipfs1_url, timeout=5.0), KuboIPFSClient(ipfs2_url, timeout=5.0)]
    if ipfs3_url:
        clients.append(KuboIPFSClient(ipfs3_url, timeout=5.0))
    peer_names = [f"ipfs-node{index + 1}" for index in range(len(clients))]
    deadline = time.time() + timeout_s
    while True:
        try:
            ids = [client.id_info() for client in clients]
            for source, source_name in zip(clients, peer_names, strict=True):
                for target_name, target in zip(peer_names, ids, strict=True):
                    if target_name == source_name:
                        continue
                    source.swarm_connect(f"/dns4/{target_name}/tcp/4001/p2p/{target['ID']}")
            peers = clients[0].swarm_peers()
            described = ", ".join(
                f"{name} ({peer_id['ID'][:12]}...)" for name, peer_id in zip(peer_names, ids, strict=True)
            )
            print(
                f"   - Kubo Private Swarm (LIBP2P_FORCE_PNET=1): {described}, peers={len(peers)}"
            )
            return
        except Exception as exc:
            if time.time() >= deadline:
                print(f"   - Kubo Private Swarm note: {exc}")
                return
            time.sleep(0.5)


def action_health(ssl_ctx, args) -> None:
    wait_for_nodes(
        ssl_ctx,
        [
            f"{args.node1_raft}/health",
            f"{args.node2_raft}/health",
            f"{args.node3_raft}/health",
            f"{args.node1_rag}/status",
            f"{args.node2_rag}/status",
            f"{args.node3_rag}/status",
        ],
    )
    for name, url in (
        ("node1-maestro", args.node1_raft),
        ("node2-peer", args.node2_raft),
        ("node3-peer", args.node3_raft),
    ):
        health = mtls_request(ssl_ctx, f"{url}/health")
        print(
            f"   - {name:<13} Raft role={health['role']}, quorum={health['quorum']}, "
            f"members={len(health['members'])}, client_cn={health['verified_client_cn']}"
        )
    connect_kubo_private_swarm(args.node1_ipfs, args.node2_ipfs, args.node3_ipfs)


def action_ingest(ssl_ctx, args) -> None:
    ring = shard_ring(args)
    urls = rag_urls(args)
    print(
        f"Ingesting {len(CORPUS)} synthetic PDFs into {ring.num_shards} hash shards "
        f"(ownership by seeded SHA-256 of doc_id) + Kubo IPFS CIDs + Cross-Node Backups over mTLS..."
    )
    for doc in CORPUS:
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
            f"{urls[assignment['owner']]}/shard/ingest-full",
            method="POST",
            payload={**doc, "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii")},
        )
        stubbed = sorted(res["peer_metadata_stubs"])
        print(
            f"   [{assignment['shard_id']} -> {assignment['owner']} | backup {assignment['backup']}] "
            f"{doc['doc_id']} pinned CID={res['manifest']['ipfs_cid'][:18]}...; "
            f"stubs on {stubbed} (Raft block #{res['raft_block_index']}, "
            f"sha256={res['manifest']['pdf_sha256'][:10]}...)"
        )
        if res["replica_warnings"]:
            print(f"     ! replica warnings: {res['replica_warnings']}")
    print(f"   - shard balance: {ring.balance(doc['doc_id'] for doc in CORPUS)}")


def action_train(ssl_ctx, args) -> None:
    print(
        f"Running {args.rounds} Federated LoRA Training Rounds "
        f"(mode={args.lora_mode}, aggregation={args.aggregation}) across all reachable members..."
    )
    for round_id in range(1, args.rounds + 1):
        round_res = mtls_request(
            ssl_ctx,
            f"{args.node1_rag}/fed/run-round",
            method="POST",
            payload={
                "round_id": round_id,
                "lora_mode": args.lora_mode,
                "aggregation": args.aggregation,
            },
        )
        per_client = " | ".join(
            f"{client['node_id']} loss {client['first_loss']:.4f}->{client['last_loss']:.4f} "
            f"({client['num_examples']} ex)"
            for client in round_res["clients"]
        )
        print(
            f"   [Round {round_id} | {args.lora_mode} | {round_res['aggregation']}, "
            f"server_lr={round_res['server_lr']}] {per_client} | "
            f"global loss={round_res['weighted_loss']:.4f} | "
            f"participants={sorted(round_res['participants'])}"
            + (
                f" | unavailable={sorted(round_res['unavailable_members'])}"
                if round_res["unavailable_members"]
                else ""
            )
            + f" | global_cid={round_res['global_adapter_cid'][:16]}... "
            f"(Raft block #{round_res['raft_block_index']})"
        )


def action_query(ssl_ctx, args) -> None:
    queries = (
        [args.query]
        if args.query
        else [
            "Which SODTYPE code identifies customer records in TRDR?",
            "What SOSOURCE code is used for purchase invoices from suppliers in FINDOC?",
            "How are credit notes recorded in FINDOC?",
        ]
    )
    print("Running Decentralized Scatter-Gather RAG Queries across all shards...")
    for query in queries:
        q_res = mtls_request(
            ssl_ctx,
            f"{args.node1_rag}/rag/query-decentralized",
            method="POST",
            payload={"query": query, "k": 2},
        )
        coverage = (
            "complete"
            if q_res["complete"]
            else f"degraded: missing {q_res['missing_shards']} ({q_res['missing_nodes']})"
        )
        top = q_res["top_k"][0]
        print(
            f"   Query: {query!r}\n"
            f"     -> {coverage} | searched={sorted(q_res['searched_nodes'])}\n"
            f"     -> Top Hit from {top['node_id']} ({top['doc_id']}): {top['text']!r}\n"
            f"     -> Cosine Score={top['score']} | On-Chain SHA-256 Verified={top['ledger_hash_verified']}"
        )


def action_tamper(ssl_ctx, args) -> None:
    print("Verifying Tamper Rejection on Raft Ledger...")
    rejected = False
    try:
        mtls_request(
            ssl_ctx,
            f"{args.node1_raft}/tx/propose",
            method="POST",
            payload={
                "tx_type": "BACKUP_CHECK",
                "payload": {
                    "doc_id": "doc-121",
                    "source_node": "node1-maestro",
                    "backup_node": "node2-peer",
                    "computed_sha256": "f" * 64,
                },
            },
        )
    except RuntimeError as exc:
        rejected = True
        print(f"   -> Tampered backup checksum properly rejected by Raft chaincode: {exc}")

    if not rejected:
        raise RuntimeError("Expected tampered backup checksum to be rejected!")


def action_status(ssl_ctx, args) -> None:
    ring = shard_ring(args)
    statuses = {
        name: mtls_request(ssl_ctx, f"{url}/status")
        for name, url in (
            ("node1-maestro", args.node1_rag),
            ("node2-peer", args.node2_rag),
            ("node3-peer", args.node3_rag),
        )
    }
    print(f"Cluster Consistency & Storage Check ({ring.num_shards} shards):")
    for name, status in statuses.items():
        store = status["rag_store"]
        shard = store["shard"]
        ledger = status["raft"]["ledger"]
        ipfs = status.get("ipfs", {})
        print(
            f"   - {name:<13} owns {shard['shard_id']:<8} "
            f"full_docs={len(store['full_shard_documents'])}, "
            f"metadata_stubs={len(store['metadata_only_stubs'])}, "
            f"verified_backups={len(store['verified_backups'])} (backup of {shard['backup_target']}), "
            f"ipfs={ipfs.get('mode')} (pins={len(ipfs.get('pins', []))}), "
            f"raft_height={ledger['height']}, tip_hash={ledger['last_hash'][:16]}..."
        )

    owners = {status["rag_store"]["shard"]["shard_id"]: status["rag_store"]["node_id"] for status in statuses.values()}
    expected = {ring.shard_id(index) for index in range(ring.num_shards)}
    assert set(owners) == expected, f"shard owners {sorted(owners)} != {sorted(expected)}"
    coverage: dict[str, int] = {}
    for status in statuses.values():
        for shard_id, count in status["rag_store"]["shard_coverage"].items():
            coverage[shard_id] = max(coverage.get(shard_id, 0), count)
    described = ", ".join(
        f"{shard_id}({owners[shard_id]})={count}" for shard_id, count in sorted(coverage.items())
    )
    print(f"   - shard coverage: {described}")

    tips = {(status["raft"]["ledger"]["height"], status["raft"]["ledger"]["last_hash"]) for status in statuses.values()}
    assert len(tips) == 1, f"Raft ledger state diverged between nodes: {sorted(tips)}"


def action_ledger(ssl_ctx, args) -> None:
    snapshot = mtls_request(ssl_ctx, f"{args.node1_raft}/ledger")
    print(json.dumps(snapshot, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-node mTLS Federated LoRA & Sharded RAG CLI")
    parser.add_argument(
        "--action",
        default="all",
        choices=("all", "health", "ingest", "train", "query", "tamper", "status", "ledger"),
    )
    parser.add_argument("--node1-id", default="node1-maestro")
    parser.add_argument("--node2-id", default="node2-peer")
    parser.add_argument("--node3-id", default="node3-peer")
    parser.add_argument("--node1-rag", default="https://localhost:18443")
    parser.add_argument("--node2-rag", default="https://localhost:28443")
    parser.add_argument("--node3-rag", default="https://localhost:38443")
    parser.add_argument("--node1-raft", default="https://localhost:17050")
    parser.add_argument("--node2-raft", default="https://localhost:27050")
    parser.add_argument("--node3-raft", default="https://localhost:37050")
    parser.add_argument("--node1-ipfs", default="http://localhost:15001")
    parser.add_argument("--node2-ipfs", default="http://localhost:25001")
    parser.add_argument("--node3-ipfs", default="http://localhost:35001")
    parser.add_argument("--connection-profile", default="./connection-profile.json")
    parser.add_argument("--certs-dir", default="./certs")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lora-mode", default="fedavg", choices=("fedavg", "ffa", "dora"))
    parser.add_argument("--aggregation", default="average", choices=("average", "summed"))
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
        print("\nSUCCESS: 3-node mTLS Raft + hash-sharded RAG + Federated LoRA test passed!")


if __name__ == "__main__":
    main()
