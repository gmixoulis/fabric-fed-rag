import base64
from types import SimpleNamespace

import pytest

from fedrag.embed import HashEmbedder
from fedrag.node_service import NodeApp


def test_ingest_rejects_mismatched_peer_before_writing(monkeypatch):
    app = NodeApp.__new__(NodeApp)
    app.store = SimpleNamespace(embedder=HashEmbedder())
    app.peer_url = "https://peer"
    app.ssl_client_ctx = None
    monkeypatch.setattr("fedrag.node_service.mtls_request", lambda *args, **kwargs: {"digest": "a" * 64})
    with pytest.raises(ValueError, match="differs between nodes"):
        app.ingest_full_and_replicate({"doc_id": "doc", "pdf_base64": base64.b64encode(b"pdf").decode()})


def test_ingest_rejects_mismatched_ledger_before_writing(monkeypatch):
    app = NodeApp.__new__(NodeApp)
    app.store = SimpleNamespace(embedder=HashEmbedder())
    app.peer_url = ""
    ledger = {"ledger": {"documents": {"doc": {"embedding_model_digest": "a" * 64}}}}
    monkeypatch.setattr(app, "get_raft_ledger", lambda: ledger)
    with pytest.raises(ValueError, match="differs from ledger"):
        app.ingest_full_and_replicate({"doc_id": "doc", "pdf_base64": base64.b64encode(b"pdf").decode()})


def test_manifest_records_embedding_digest(monkeypatch):
    app = NodeApp.__new__(NodeApp)
    embedder = HashEmbedder()
    app.store = SimpleNamespace(embedder=embedder, ingest_full_document=lambda **kwargs: {"doc_id": "doc"})
    app.ipfs = SimpleNamespace(add_bytes=lambda *args, **kwargs: "cid")
    app.node_id = "node1"
    app.peer_url = ""
    monkeypatch.setattr(app, "get_raft_ledger", lambda: {"ledger": {"documents": {}}})
    recorded = {}

    def propose(tx_type, payload):
        recorded.update(payload)
        return {"index": 1, "entry_hash": "entry"}

    monkeypatch.setattr(app, "propose_raft_tx", propose)
    app.ingest_full_and_replicate({
        "doc_id": "doc", "title": "test", "chunks": [],
        "pdf_base64": base64.b64encode(b"pdf").decode(),
    })
    assert recorded["embedding_model_digest"] == embedder.digest
