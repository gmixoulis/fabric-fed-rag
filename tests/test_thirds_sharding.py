"""Offline WP3 tests: hash shards in thirds, ring backups, three-peer scatter-gather, 3-member rounds.

Everything runs on ephemeral localhost TLS 1.3 mTLS sockets, temporary directories, the
deterministic `HashEmbedder`, and temporary ledger files. No Docker, no GPU, no network.
"""

from __future__ import annotations

import base64
import socket
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from fedrag.certgen import generate_certs
from fedrag.embed import HashEmbedder
from fedrag.fedavg import aggregate_adapters, average_adapters, server_lr_for
from fedrag.node_service import (
    NodeApp,
    build_handler,
    make_client_ssl_context,
    make_server_ssl_context,
    mtls_request,
)
from fedrag.pdf_embedder import build_full_pdf_with_metadata
from fedrag.raft_ledger import LedgerState, RaftNode, build_raft_handler
from fedrag.shard_store import ShardedRAGStore, ShardRing, shard_index_for

NODE_IDS = ["node1-maestro", "node2-peer", "node3-peer"]
LEADER_ID = "node1-maestro"
DOC_IDS = [f"doc-{index}" for index in range(121, 130)]  # 3 per shard at N=3, 4/5 at N=2


def _corpus() -> list[dict]:
    return [
        {
            "doc_id": doc_id,
            "title": f"Synthetic ERP specification {doc_id}",
            "page_count": 2,
            "chunks": [
                f"Document {doc_id} covers SODTYPE code {100 + offset} in table TRDR. "
                + " ".join(f"filler{offset} note{index}" for index in range(offset + 2)),
                f"Document {doc_id} also describes SOSOURCE {2000 + offset} in FINDOC.",
            ],
            "training_pairs": [
                {
                    "prompt": f"Which SODTYPE code does {doc_id} describe?",
                    "target": f"SELECT {100 + offset} FROM TRDR WHERE SODTYPE = {100 + offset};",
                }
            ],
        }
        for offset, doc_id in enumerate(DOC_IDS)
    ]


def _pdf(doc: dict, owner: str, backup: str) -> bytes:
    pdf_bytes, _meta = build_full_pdf_with_metadata(
        doc_id=doc["doc_id"],
        title=doc["title"],
        shard_owner=owner,
        metadata_holder=backup,
        chunks=doc["chunks"],
        page_count=doc["page_count"],
    )
    return pdf_bytes


# --------------------------------------------------------------- shard ring ---
def test_shard_assignment_is_deterministic_and_balanced():
    for members, expected_top in ((["node1-maestro", "node2-peer"], 5), (NODE_IDS, 3)):
        ring = ShardRing(members)
        again = ShardRing(reversed(members))  # member order in the list must not matter
        assert ring.members == again.members
        assignments = [ring.index(doc_id) for doc_id in DOC_IDS]
        assert assignments == [again.index(doc_id) for doc_id in DOC_IDS]
        assert assignments == [shard_index_for(doc_id, len(members)) for doc_id in DOC_IDS]

        counts = ring.balance(DOC_IDS)
        assert sum(counts.values()) == len(DOC_IDS)
        assert max(counts.values()) - min(counts.values()) <= 1, counts
        assert max(counts.values()) == expected_top, counts

    # A different salt is a different (also deterministic) split.
    other = ShardRing(NODE_IDS, salt="fabric-fed-rag/shard-ring/v2")
    assert [other.index(doc_id) for doc_id in DOC_IDS] != [ShardRing(NODE_IDS).index(doc_id) for doc_id in DOC_IDS]


def test_every_document_has_one_owner_and_one_backup_on_another_node():
    for members in (["node1-maestro", "node2-peer"], NODE_IDS):
        ring = ShardRing(members)
        assert ring.num_shards == len(members)
        owners: dict[str, str] = {}
        for doc_id in DOC_IDS:
            assignment = ring.assignment(doc_id)
            assert assignment["owner"] in members
            assert assignment["backup"] in members
            assert assignment["backup"] != assignment["owner"]
            assert ring.shard_id(assignment["shard_index"]) == assignment["shard_id"]
            owners[doc_id] = assignment["owner"]
            assert ring.stub_holders(doc_id) == [m for m in members if m != assignment["owner"]]
        # Exactly one owner per document: the assignments partition the corpus.
        assert sorted(owners) == sorted(DOC_IDS)

    # Ring order: 1 backs up 2, 2 backs up 3, 3 backs up 1.
    ring = ShardRing(NODE_IDS)
    backups = {ring.shard_id(index): ring.backup_of_index(index) for index in range(3)}
    assert backups == {"shard-1": "node2-peer", "shard-2": "node3-peer", "shard-3": "node1-maestro"}


def test_single_member_ring_refuses_a_backup():
    with pytest.raises(ValueError, match="at least one member"):
        ShardRing([])
    with pytest.raises(ValueError, match="at least two ring members"):
        ShardRing(["node1-maestro"]).backup("doc-121")


# ------------------------------------------- stores: metadata + backup restore ---
def _populate_ring_stores(tmp_path, members: list[str]) -> dict[str, ShardedRAGStore]:
    embedder = HashEmbedder()
    ring = ShardRing(members)
    stores = {
        node_id: ShardedRAGStore(node_id, tmp_path / node_id, embedder, members=members)
        for node_id in members
    }
    for doc in _corpus():
        owner = ring.owner(doc["doc_id"])
        backup = ring.backup(doc["doc_id"])
        manifest = stores[owner].ingest_full_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            pdf_bytes=_pdf(doc, owner, backup),
            chunks=doc["chunks"],
            page_count=doc["page_count"],
            training_pairs=doc["training_pairs"],
        )
        for holder in ring.stub_holders(doc["doc_id"]):
            stores[holder].ingest_metadata_stub(manifest)
        stores[backup].store_and_verify_backup(
            doc_id=doc["doc_id"],
            source_node=owner,
            pdf_bytes=_pdf(doc, owner, backup),
            expected_sha256=manifest["pdf_sha256"],
        )
    return stores


def test_losing_one_node_keeps_every_document_metadata(tmp_path):
    stores = _populate_ring_stores(tmp_path, NODE_IDS)
    for node_id, store in stores.items():
        rows = store.all_document_metadata()
        assert sorted(row["doc_id"] for row in rows) == sorted(DOC_IDS), node_id
        owned = [row for row in rows if not row["is_metadata_only"]]
        stubbed = [row for row in rows if row["is_metadata_only"]]
        assert len(owned) == 3 and len(stubbed) == 6, node_id
        assert all(row["shard_owner"] == node_id for row in owned)

    # Any one node lost: both survivors still answer metadata for every document.
    for lost in NODE_IDS:
        survivors = [node_id for node_id in NODE_IDS if node_id != lost]
        for survivor in survivors:
            known = {row["doc_id"] for row in stores[survivor].all_document_metadata()}
            assert known == set(DOC_IDS), (lost, survivor)


def test_backup_replicas_restore_the_lost_shard(tmp_path):
    stores = _populate_ring_stores(tmp_path, NODE_IDS)
    ring = ShardRing(NODE_IDS)
    lost = "node2-peer"
    backup_holder = ring.backup_of_index(ring.index_of_node(lost))
    assert backup_holder == "node3-peer"
    owned_by_lost = [doc["doc_id"] for doc in _corpus() if ring.owner(doc["doc_id"]) == lost]
    assert owned_by_lost

    # The shard owner is wiped and rebuilt from the verified replicas held by its successor.
    restored = ShardedRAGStore(lost, tmp_path / "node2-restored", HashEmbedder(), members=NODE_IDS)
    for doc_id in owned_by_lost:
        replica = stores[backup_holder].read_backup(doc_id)
        row = restored.ingest_restored_backup(replica)
        original = next(row for row in stores[lost].all_document_metadata() if row["doc_id"] == doc_id)
        assert row["pdf_sha256"] == original["pdf_sha256"]
        assert row["chunks_sha256"] == original["chunks_sha256"]
        assert row["content_sha256"] == original["content_sha256"]
        assert row["shard_owner"] == lost

    restored_rows = {row["doc_id"]: row for row in restored.all_document_metadata()}
    assert sorted(restored_rows) == sorted(owned_by_lost)
    assert all(row["is_metadata_only"] == 0 for row in restored_rows.values())
    hits = restored.search_local("SODTYPE code", k=3)
    assert hits and all(hit["node_id"] == lost for hit in hits)

    # A corrupted replica is refused instead of being restored silently.
    replica = stores[backup_holder].read_backup(owned_by_lost[0])
    tampered = {**replica, "pdf_bytes": replica["pdf_bytes"].replace(b"SODTYPE", b"SODTYPX")}
    with pytest.raises(ValueError, match="PDF binary SHA-256 mismatch"):
        restored.ingest_restored_backup(tampered)


# --------------------------------------------- aggregation modes (unit, no TLS) ---
def test_summed_updates_are_a_server_learning_rate_of_the_participants(tmp_path):
    from fedrag.trainer import init_lora_state

    previous = init_lora_state()
    first = {key: value + np.float32(0.5) for key, value in previous.items()}
    second = {key: value + np.float32(0.5) for key, value in previous.items()}
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    averaged = average_adapters([(first, 3.0), (second, 3.0)])
    assert np.allclose(averaged[key], previous[key] + np.float32(0.5))

    plain = aggregate_adapters([(first, 3.0), (second, 3.0)])
    for key in plain:
        assert np.array_equal(plain[key], averaged[key])  # default stays plain averaging

    summed = aggregate_adapters([(first, 3.0), (second, 3.0)], previous=previous, server_lr=server_lr_for("summed", 2))
    for key in summed:
        assert np.allclose(
            summed[key], previous[key] + np.float32(2.0) * (averaged[key] - previous[key])
        )

    assert server_lr_for("average", 3) == 1.0
    assert server_lr_for("summed", 3) == 3.0
    with pytest.raises(ValueError, match="unknown aggregation"):
        server_lr_for("geometric", 3)
    with pytest.raises(ValueError, match="needs the adapter"):
        aggregate_adapters([(first, 3.0)], server_lr=2.0)


# ------------------------------------------------- in-process mTLS cluster ---
def _start_tls_server(handler_cls, server_ssl_ctx) -> tuple[ThreadingHTTPServer, int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    httpd.socket = server_ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    certs_dir = tmp_path_factory.mktemp("wp3-certs") / "certs"
    generate_certs(certs_dir, NODE_IDS, ["127.0.0.1"])
    return certs_dir


def _build_cluster(tmp_path, certs_dir):
    ca_file = str(certs_dir / "ca.crt")
    servers = {}
    rafts: dict[str, RaftNode] = {}
    apps: dict[str, NodeApp] = {}

    def cert_paths(node_id: str) -> tuple[str, str]:
        return str(certs_dir / f"{node_id}.crt"), str(certs_dir / f"{node_id}.key")

    for node_id in NODE_IDS:
        cert, key = cert_paths(node_id)
        raft = RaftNode(
            node_id=node_id,
            role="follower",
            leader_url="https://127.0.0.1:1",
            peer_urls=[],
            ssl_client_ctx=make_client_ssl_context(ca_file, cert, key),
            ledger=LedgerState(tmp_path / node_id / "ledger.json", members=NODE_IDS),
            members={node_id: "https://127.0.0.1:1"},
            election_timeout=60.0,
        )
        servers[f"raft-{node_id}"], port = _start_tls_server(
            build_raft_handler(raft), make_server_ssl_context(ca_file, cert, key)
        )
        raft.port = port
        rafts[node_id] = raft

    raft_urls = {node_id: f"https://127.0.0.1:{rafts[node_id].port}" for node_id in NODE_IDS}
    for node_id, raft in rafts.items():
        raft.members = dict(raft_urls)
        raft.leader_url = raft_urls[LEADER_ID]
        raft.leader_id = LEADER_ID
        if node_id == LEADER_ID:
            raft.role = "leader"

    placeholder = {node_id: "" for node_id in NODE_IDS}
    for node_id in NODE_IDS:
        cert, key = cert_paths(node_id)
        app = NodeApp(
            node_id=node_id,
            data_dir=tmp_path / node_id / "app",
            raft_url=raft_urls[node_id],
            members=placeholder,
            ca_file=ca_file,
            cert_file=cert,
            key_file=key,
            embedder=HashEmbedder(),
            peer_timeout=5.0,
            train_timeout=60.0,
        )
        servers[f"app-{node_id}"], port = _start_tls_server(
            build_handler(app), make_server_ssl_context(ca_file, cert, key)
        )
        app.port = port
        apps[node_id] = app

    app_urls = {node_id: f"https://127.0.0.1:{apps[node_id].port}" for node_id in NODE_IDS}
    for app in apps.values():
        app.configure_member_urls(app_urls)

    return {"servers": servers, "rafts": rafts, "apps": apps, "app_urls": app_urls}


def _stop_server(cluster: dict, key: str) -> None:
    server = cluster["servers"].pop(key, None)
    if server is not None:
        server.shutdown()
        server.server_close()


def _stop_cluster(cluster: dict) -> None:
    for key in list(cluster["servers"]):
        _stop_server(cluster, key)


@pytest.fixture
def cluster(tmp_path, certs):
    built = _build_cluster(tmp_path, certs)
    try:
        yield built
    finally:
        _stop_cluster(built)


def _ssl_ctx(certs_dir, node_id: str = LEADER_ID):
    return make_client_ssl_context(
        str(certs_dir / "ca.crt"),
        str(certs_dir / f"{node_id}.crt"),
        str(certs_dir / f"{node_id}.key"),
    )


def _ingest_corpus(ssl_ctx, cluster) -> dict[str, dict]:
    ring = ShardRing(NODE_IDS)
    manifests = {}
    for doc in _corpus():
        owner = ring.owner(doc["doc_id"])
        pdf_bytes = _pdf(doc, owner, ring.backup(doc["doc_id"]))
        res = mtls_request(
            ssl_ctx,
            f"{cluster['app_urls'][owner]}/shard/ingest-full",
            method="POST",
            payload={**doc, "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii")},
        )
        assert res["shard_assignment"]["owner"] == owner
        assert res["peer_backup_attestation"]["verified"] is True
        assert res["replica_warnings"] == []
        manifests[doc["doc_id"]] = res["manifest"]
    return manifests


def _single_index(tmp_path) -> ShardedRAGStore:
    """One node holding the whole corpus, the reference for scatter-gather recall."""
    store = ShardedRAGStore("reference-index", tmp_path / "reference", HashEmbedder())
    for doc in _corpus():
        store.ingest_full_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            pdf_bytes=_pdf(doc, "reference-index", "reference-index"),
            chunks=doc["chunks"],
        )
    return store


def test_three_peer_scatter_gather_equals_single_index_and_degrades(cluster, certs, tmp_path):
    ssl_ctx = _ssl_ctx(certs)
    _ingest_corpus(ssl_ctx, cluster)
    query = "SODTYPE code 105 in TRDR"
    k = 3

    reference = _single_index(tmp_path)
    expected = [
        (hit["doc_id"], hit["chunk_id"], hit["score"]) for hit in reference.search_local(query, k=k)
    ]

    full = mtls_request(
        ssl_ctx,
        f"{cluster['app_urls'][LEADER_ID]}/rag/query-decentralized",
        method="POST",
        payload={"query": query, "k": k},
    )
    assert full["complete"] is True
    assert sorted(full["searched_nodes"]) == sorted(NODE_IDS)
    assert full["missing_nodes"] == [] and full["missing_shards"] == []
    assert [(hit["doc_id"], hit["chunk_id"], hit["score"]) for hit in full["top_k"]] == expected

    # Every node knows the whole corpus metadata (own shard full, other two shards as stubs).
    for node_id in NODE_IDS:
        status = mtls_request(ssl_ctx, f"{cluster['app_urls'][node_id]}/status")
        rows = status["rag_store"]
        assert len(rows["full_shard_documents"]) == 3
        assert len(rows["metadata_only_stubs"]) == 6
        assert len(rows["verified_backups"]) == 3
        assert rows["shard"]["shard_id"] == ShardRing(NODE_IDS).shard_id(
            ShardRing(NODE_IDS).index_of_node(node_id)
        )

    # One node down: the query still answers, reports the missing shard, and loses only its hits.
    _stop_server(cluster, "app-node3-peer")
    degraded = mtls_request(
        ssl_ctx,
        f"{cluster['app_urls'][LEADER_ID]}/rag/query-decentralized",
        method="POST",
        payload={"query": query, "k": k},
    )
    assert degraded["complete"] is False
    assert degraded["missing_nodes"] == ["node3-peer"]
    assert degraded["missing_shards"] == ["shard-3"]
    assert sorted(degraded["searched_nodes"]) == ["node1-maestro", "node2-peer"]
    assert degraded["top_k"], "a missing node must lower recall, not fail the query"
    assert all(hit["node_id"] != "node3-peer" for hit in degraded["top_k"])


def test_scatter_gather_times_out_a_hung_peer(cluster, certs, tmp_path):
    ssl_ctx = _ssl_ctx(certs)
    _ingest_corpus(ssl_ctx, cluster)

    # A peer that accepts the TCP connection but never completes the TLS handshake.
    silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    try:
        app = cluster["apps"][LEADER_ID]
        app.peer_timeout = 0.5
        app.configure_member_urls(
            {**cluster["app_urls"], "node3-peer": f"https://127.0.0.1:{silent.getsockname()[1]}"}
        )
        result = mtls_request(
            ssl_ctx,
            f"{cluster['app_urls'][LEADER_ID]}/rag/query-decentralized",
            method="POST",
            payload={"query": "SODTYPE code 105 in TRDR", "k": 2},
        )
        assert result["complete"] is False
        assert result["missing_nodes"] == ["node3-peer"]
        assert result["missing_shards"] == ["shard-3"]
        assert result["top_k"]
    finally:
        silent.close()


def test_federated_round_with_three_members_then_two_of_three(cluster, certs, tmp_path):
    ssl_ctx = _ssl_ctx(certs)
    _ingest_corpus(ssl_ctx, cluster)
    leader_app = cluster["app_urls"][LEADER_ID]

    round_one = mtls_request(
        ssl_ctx,
        f"{leader_app}/fed/run-round",
        method="POST",
        payload={"round_id": 1, "lora_mode": "fedavg"},
    )
    assert sorted(round_one["participants"]) == sorted(NODE_IDS)
    assert round_one["aggregation"] == "average" and round_one["server_lr"] == 1.0
    assert round_one["unavailable_members"] == {}
    assert round_one["total_examples"] == len(_corpus())
    assert len(round_one["clients"]) == 3

    # Summed updates: the server learning rate is the number of participants.
    round_two = mtls_request(
        ssl_ctx,
        f"{leader_app}/fed/run-round",
        method="POST",
        payload={"round_id": 2, "lora_mode": "fedavg", "aggregation": "summed"},
    )
    assert round_two["server_lr"] == 3.0
    assert round_two["weighted_loss"] < round_one["weighted_loss"]
    assert len(round_two["participants"]) == 3

    # One member missing: two of three still train and finalize at the ledger quorum.
    _stop_server(cluster, "app-node3-peer")
    round_three = mtls_request(
        ssl_ctx,
        f"{leader_app}/fed/run-round",
        method="POST",
        payload={"round_id": 3, "lora_mode": "fedavg"},
    )
    assert sorted(round_three["participants"]) == ["node1-maestro", "node2-peer"]
    assert sorted(round_three["unavailable_members"]) == ["node3-peer"]
    assert round_three["server_lr"] == 1.0

    leader_raft = f"https://127.0.0.1:{cluster['rafts'][LEADER_ID].port}"
    ledger = mtls_request(ssl_ctx, f"{leader_raft}/ledger")["ledger"]
    finalized = ledger["final_rounds"]["3"]
    assert sorted(finalized["participants"]) == ["node1-maestro", "node2-peer"]
    assert finalized["unavailable_members"] == ["node3-peer"]
    assert ledger["lora_quorum"] == 2

    # Two members missing: below quorum, the round is refused and nothing is finalized.
    _stop_server(cluster, "app-node2-peer")
    with pytest.raises(RuntimeError, match="needs 2"):
        mtls_request(
            ssl_ctx,
            f"{leader_app}/fed/run-round",
            method="POST",
            payload={"round_id": 4, "lora_mode": "fedavg"},
        )
    ledger_after = mtls_request(ssl_ctx, f"{leader_raft}/ledger")["ledger"]
    assert "4" not in ledger_after["final_rounds"]


def test_backup_export_and_restore_over_mtls(cluster, certs):
    ssl_ctx = _ssl_ctx(certs)
    manifests = _ingest_corpus(ssl_ctx, cluster)
    ring = ShardRing(NODE_IDS)

    owned_by_second = next(doc_id for doc_id in DOC_IDS if ring.owner(doc_id) == "node2-peer")
    second = cluster["app_urls"]["node2-peer"]
    third = cluster["app_urls"]["node3-peer"]

    exported = mtls_request(
        ssl_ctx, f"{third}/shard/export-backup", method="POST", payload={"doc_id": owned_by_second}
    )
    assert exported["source_node"] == "node2-peer"
    assert exported["pdf_sha256"] == manifests[owned_by_second]["pdf_sha256"]

    restored = mtls_request(
        ssl_ctx, f"{second}/shard/restore", method="POST", payload={"doc_ids": [owned_by_second]}
    )
    assert restored["backup_node"] == "node3-peer"
    assert restored["restored_documents"] == 1 and restored["failures"] == {}
    row = restored["restored"][0]
    assert row["pdf_sha256"] == manifests[owned_by_second]["pdf_sha256"]
    assert row["chunks_sha256"] == manifests[owned_by_second]["chunks_sha256"]

    # Without doc_ids the node restores everything the ledger says it owns.
    everything = mtls_request(ssl_ctx, f"{second}/shard/restore", method="POST", payload={})
    owned = sorted(doc_id for doc_id in DOC_IDS if ring.owner(doc_id) == "node2-peer")
    assert sorted(r["doc_id"] for r in everything["restored"]) == owned
    assert everything["failures"] == {}

    # A member that is not the ring successor refuses to hold a second replica.
    first = cluster["app_urls"]["node1-maestro"]
    with pytest.raises(RuntimeError, match="is backed up by node3-peer"):
        mtls_request(
            ssl_ctx,
            f"{first}/shard/store-backup",
            method="POST",
            payload={
                "doc_id": owned_by_second,
                "source_node": "node2-peer",
                "pdf_base64": base64.b64encode(b"%PDF-1.4 not the committed bytes %%EOF").decode("ascii"),
                "expected_sha256": manifests[owned_by_second]["pdf_sha256"],
            },
        )
