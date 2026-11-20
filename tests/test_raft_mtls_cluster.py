"""In-process 2-node TLS 1.3 mTLS cluster test: Raft replication, two-way sharding, Federated LoRA.

This is the N = 2 regression for the N-way shard ring: two members must still split the
corpus in half, each backing up the other's shard, exactly as before WP3.
"""

from __future__ import annotations

import base64
import threading
from http.server import ThreadingHTTPServer

import pytest

from fedrag.certgen import generate_certs
from fedrag.embed import HashEmbedder
from fedrag.node_service import (
    NodeApp,
    build_handler,
    make_client_ssl_context,
    make_server_ssl_context,
    mtls_request,
)
from fedrag.raft_ledger import LedgerState, RaftNode, build_raft_handler


def _start_tls_server(handler_cls, server_ssl_ctx) -> tuple[ThreadingHTTPServer, int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    httpd.socket = server_ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def test_two_node_mtls_raft_and_federated_rag_cluster(tmp_path):
    certs_dir = tmp_path / "certs"
    generate_certs(certs_dir, ["node1-maestro", "node2-peer"], ["127.0.0.1"])

    ca_file = str(certs_dir / "ca.crt")
    n1_cert = str(certs_dir / "node1-maestro.crt")
    n1_key = str(certs_dir / "node1-maestro.key")
    n2_cert = str(certs_dir / "node2-peer.crt")
    n2_key = str(certs_dir / "node2-peer.key")

    n1_client_ctx = make_client_ssl_context(ca_file, n1_cert, n1_key)
    n2_client_ctx = make_client_ssl_context(ca_file, n2_cert, n2_key)
    n1_server_ctx = make_server_ssl_context(ca_file, n1_cert, n1_key)
    n2_server_ctx = make_server_ssl_context(ca_file, n2_cert, n2_key)

    # 1. Start Follower Raft node (node2-peer) first to get its port
    ledger2 = LedgerState(tmp_path / "node2" / "ledger.json")
    raft2 = RaftNode(
        node_id="node2-peer",
        role="follower",
        leader_url="https://127.0.0.1:0",  # updated below
        peer_urls=[],
        ssl_client_ctx=n2_client_ctx,
        ledger=ledger2,
    )
    raft2_srv, raft2_port = _start_tls_server(build_raft_handler(raft2), n2_server_ctx)

    # 2. Start Leader Raft node (node1-maestro) pointing to follower
    ledger1 = LedgerState(tmp_path / "node1" / "ledger.json")
    raft1 = RaftNode(
        node_id="node1-maestro",
        role="leader",
        leader_url="https://127.0.0.1:0",
        peer_urls=[f"https://127.0.0.1:{raft2_port}"],
        ssl_client_ctx=n1_client_ctx,
        ledger=ledger1,
    )
    raft1_srv, raft1_port = _start_tls_server(build_raft_handler(raft1), n1_server_ctx)
    raft1.leader_url = f"https://127.0.0.1:{raft1_port}"
    raft2.leader_url = f"https://127.0.0.1:{raft1_port}"

    # 3. Start NodeApp RAG + Trainer servers for both nodes. The member list defines the
    # shard ring; the URLs are published once both ephemeral ports are known.
    member_ids = {"node1-maestro": "", "node2-peer": ""}
    app2 = NodeApp(
        node_id="node2-peer",
        data_dir=tmp_path / "node2",
        raft_url=f"https://127.0.0.1:{raft2_port}",
        members=member_ids,
        ca_file=ca_file,
        cert_file=n2_cert,
        key_file=n2_key,
        embedder=HashEmbedder(),
    )
    app2_srv, app2_port = _start_tls_server(build_handler(app2), n2_server_ctx)

    app1 = NodeApp(
        node_id="node1-maestro",
        data_dir=tmp_path / "node1",
        raft_url=f"https://127.0.0.1:{raft1_port}",
        members=member_ids,
        ca_file=ca_file,
        cert_file=n1_cert,
        key_file=n1_key,
        embedder=HashEmbedder(),
    )
    app1_srv, app1_port = _start_tls_server(build_handler(app1), n1_server_ctx)

    urls = {
        "node1-maestro": f"https://127.0.0.1:{app1_port}",
        "node2-peer": f"https://127.0.0.1:{app2_port}",
    }
    app1.configure_member_urls(urls)
    app2.configure_member_urls(urls)
    assert app1.shard_id == "shard-1" and app2.shard_id == "shard-2"
    assert app1.store.backup_target() == "node2-peer"
    assert app2.store.backup_target() == "node1-maestro"

    try:
        # Verify mTLS health and client certificate CN
        h1 = mtls_request(n1_client_ctx, f"https://127.0.0.1:{raft1_port}/health")
        assert h1["verified_client_cn"] == "node1-maestro"

        # Ingest Shard 1 document on node1-maestro
        pdf1 = b"%PDF-1.4 Shard 1 PDF %%EOF"
        res1 = mtls_request(
            n1_client_ctx,
            f"https://127.0.0.1:{app1_port}/shard/ingest-full",
            method="POST",
            payload={
                "doc_id": "doc-a",
                "title": "Customer Spec",
                "pdf_base64": base64.b64encode(pdf1).decode("ascii"),
                "chunks": ["Customer records use SODTYPE 13 in TRDR."],
                "page_count": 1,
                "training_pairs": [
                    {
                        "prompt": "Count customers",
                        "target": "SELECT COUNT(*) FROM TRDR WHERE SODTYPE = 13;",
                    }
                ],
            },
        )
        assert res1["shard_assignment"]["owner"] == "node1-maestro"
        assert res1["shard_assignment"]["backup"] == "node2-peer"
        assert res1["peer_metadata_stub"]["is_metadata_only"] is True
        assert res1["peer_backup_attestation"]["verified"] is True

        # Ingest Shard 2 document on node2-peer (follower forwards tx to leader over mTLS)
        pdf2 = b"%PDF-1.4 Shard 2 PDF %%EOF"
        res2 = mtls_request(
            n1_client_ctx,
            f"https://127.0.0.1:{app2_port}/shard/ingest-full",
            method="POST",
            payload={
                "doc_id": "doc-d",
                "title": "Purchase Spec",
                "pdf_base64": base64.b64encode(pdf2).decode("ascii"),
                "chunks": ["Supplier purchase invoices use SOSOURCE 1251 in FINDOC."],
                "page_count": 1,
                "training_pairs": [
                    {
                        "prompt": "Count purchase invoices",
                        "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1251;",
                    }
                ],
            },
        )
        assert res2["shard_assignment"]["owner"] == "node2-peer"
        assert res2["shard_assignment"]["backup"] == "node1-maestro"
        assert res2["peer_metadata_stub"]["is_metadata_only"] is True
        assert res2["peer_backup_attestation"]["verified"] is True

        # Run 2 rounds of FFA-LoRA federated training across both nodes
        r1 = mtls_request(
            n1_client_ctx,
            f"https://127.0.0.1:{app1_port}/fed/run-round",
            method="POST",
            payload={"round_id": 1, "lora_mode": "ffa"},
        )
        r2 = mtls_request(
            n1_client_ctx,
            f"https://127.0.0.1:{app1_port}/fed/run-round",
            method="POST",
            payload={"round_id": 2, "lora_mode": "ffa"},
        )
        assert r2["weighted_loss"] < r1["weighted_loss"]

        # Run scatter-gather RAG query across both shards
        q_res = mtls_request(
            n1_client_ctx,
            f"https://127.0.0.1:{app1_port}/rag/query-decentralized",
            method="POST",
            payload={"query": "Supplier purchase invoices SOSOURCE 1251", "k": 2},
        )
        assert q_res["complete"] is True
        assert q_res["missing_shards"] == []
        assert q_res["top_k"][0]["doc_id"] == "doc-d"
        assert q_res["top_k"][0]["node_id"] == "node2-peer"
        assert q_res["top_k"][0]["ledger_hash_verified"] is True

        # Confirm tampered backup check is rejected
        with pytest.raises(RuntimeError, match="Backup checksum mismatch"):
            mtls_request(
                n1_client_ctx,
                f"https://127.0.0.1:{raft1_port}/tx/propose",
                method="POST",
                payload={
                    "tx_type": "BACKUP_CHECK",
                    "payload": {
                        "doc_id": "doc-a",
                        "source_node": "node1-maestro",
                        "backup_node": "node2-peer",
                        "computed_sha256": "0" * 64,
                    },
                },
            )
    finally:
        app1_srv.shutdown()
        app2_srv.shutdown()
        raft1_srv.shutdown()
        raft2_srv.shutdown()
