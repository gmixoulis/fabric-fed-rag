"""WP2 embedding-digest agreement, now over the member list (node1, node2, node3).

No TLS socket and no Ollama: the peer call is patched and the shard ring is real.
"""

import base64
from types import SimpleNamespace

import pytest

from fedrag.embed import HashEmbedder
from fedrag.node_service import NodeApp
from fedrag.shard_store import ShardRing

MEMBERS = ["node1-maestro", "node2-peer", "node3-peer"]


def _peer_app(monkeypatch, peer_digests: dict[str, str]) -> NodeApp:
    """A NodeApp shell with a real shard ring and a patched mTLS peer call."""
    app = NodeApp.__new__(NodeApp)
    app.node_id = "node1-maestro"
    app.store = SimpleNamespace(embedder=HashEmbedder(), ring=ShardRing(MEMBERS))
    app.members = {member: f"https://{member}" for member in MEMBERS}
    app.peers = {member: f"https://{member}" for member in MEMBERS if member != app.node_id}
    app.ssl_client_ctx = None
    app.peer_timeout = 5.0
    app.get_raft_ledger = lambda: {"ledger": {"documents": {}}}

    def fake_mtls_request(_ctx, url, method="GET", payload=None, timeout=30.0):
        peer_id = url.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0]
        if method == "GET" and peer_id in peer_digests:
            return {"digest": peer_digests[peer_id]}
        raise RuntimeError(f"unreachable: {url}")

    monkeypatch.setattr("fedrag.node_service.mtls_request", fake_mtls_request)
    return app


def test_ingest_rejects_mismatched_peer_before_writing(monkeypatch):
    app = _peer_app(
        monkeypatch,
        {"node2-peer": HashEmbedder().digest, "node3-peer": "a" * 64},
    )
    with pytest.raises(ValueError, match="differs between nodes"):
        app.ingest_full_and_replicate(
            {"doc_id": "doc-a", "pdf_base64": base64.b64encode(b"pdf").decode()}
        )


def test_ingest_accepts_an_unreachable_third_node(monkeypatch):
    app = _peer_app(monkeypatch, {"node2-peer": HashEmbedder().digest})
    app.peers = {member: f"https://{member}" for member in MEMBERS if member != app.node_id}
    agreement = app.check_embedding_agreement()
    assert agreement["node1-maestro"]["status"] == "local"
    assert agreement["node2-peer"]["status"] == "match"
    assert agreement["node3-peer"]["status"] == "unreachable"


def test_ingest_rejects_mismatched_ledger_before_writing(monkeypatch):
    app = _peer_app(monkeypatch, {"node2-peer": HashEmbedder().digest})
    ledger = {"ledger": {"documents": {"doc": {"embedding_model_digest": "a" * 64}}}}
    monkeypatch.setattr(app, "get_raft_ledger", lambda: ledger)
    with pytest.raises(ValueError, match="differs from ledger"):
        app.ingest_full_and_replicate(
            {"doc_id": "doc-a", "pdf_base64": base64.b64encode(b"pdf").decode()}
        )


def test_manifest_records_embedding_digest(monkeypatch):
    app = NodeApp.__new__(NodeApp)
    embedder = HashEmbedder()
    app.node_id = "node1-maestro"
    app.store = SimpleNamespace(
        embedder=embedder,
        ring=ShardRing(["node1-maestro"]),  # single-member ring: this node owns every document
        ingest_full_document=lambda **kwargs: {"doc_id": kwargs["doc_id"]},
    )
    app.members = {"node1-maestro": "https://node1-maestro"}
    app.peers = {}
    app.ipfs = SimpleNamespace(add_bytes=lambda *args, **kwargs: "cid")
    app.ssl_client_ctx = None
    monkeypatch.setattr(app, "get_raft_ledger", lambda: {"ledger": {"documents": {}}})
    recorded = {}

    def propose(tx_type, payload):
        recorded.update(payload)
        return {"index": 1, "entry_hash": "entry"}

    monkeypatch.setattr(app, "propose_raft_tx", propose)
    res = app.ingest_full_and_replicate({
        "doc_id": "doc", "title": "test", "chunks": [],
        "pdf_base64": base64.b64encode(b"pdf").decode(),
    })
    assert recorded["embedding_model_digest"] == embedder.digest
    assert res["peer_backup_attestation"] is None
    assert res["replica_warnings"] == ["no backup replica: the ring has a single member"]
