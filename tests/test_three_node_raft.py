"""Offline three-member Raft tests over ephemeral localhost TLS 1.3 mTLS sockets (WP1)."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

import pytest

from fedrag.certgen import generate_certs
from fedrag.node_service import make_client_ssl_context, make_server_ssl_context
from fedrag.raft_ledger import LedgerState, RaftNode, build_raft_handler, majority, parse_member_urls

NODE_IDS = ["node1-maestro", "node2-peer", "node3-peer"]
LEADER_ID = "node1-maestro"
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    certs_dir = tmp_path_factory.mktemp("three-node-certs") / "certs"
    generate_certs(certs_dir, NODE_IDS, ["127.0.0.1"])
    return certs_dir


def _start_tls_server(handler_cls, server_ssl_ctx) -> tuple[ThreadingHTTPServer, int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    httpd.socket = server_ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _stop_server(httpd: ThreadingHTTPServer) -> None:
    httpd.shutdown()
    httpd.server_close()


def _new_raft(
    node_id: str,
    *,
    tmp_path,
    certs_dir,
    members: dict[str, str] | None = None,
    ledger_members: list[str] | None = None,
    role: str = "follower",
    leader_url: str = "https://127.0.0.1:1",
) -> tuple[RaftNode, LedgerState]:
    ca_file = str(certs_dir / "ca.crt")
    cert = str(certs_dir / f"{node_id}.crt")
    key = str(certs_dir / f"{node_id}.key")
    ledger = LedgerState(
        tmp_path / node_id / "ledger.json",
        members=ledger_members if ledger_members is not None else (list(members) if members else None),
    )
    raft = RaftNode(
        node_id=node_id,
        role=role,
        leader_url=leader_url,
        peer_urls=[],
        ssl_client_ctx=make_client_ssl_context(ca_file, cert, key),
        ledger=ledger,
        members=members,
        election_timeout=60.0,
    )
    return raft, ledger


def _build_three_node_cluster(tmp_path, certs_dir):
    """Start all three Raft servers and then publish the real membership URLs."""
    servers: dict[str, ThreadingHTTPServer] = {}
    nodes: dict[str, RaftNode] = {}
    ledgers: dict[str, LedgerState] = {}
    ports: dict[str, int] = {}
    server_ctxs = {}

    for node_id in NODE_IDS:
        ca_file = str(certs_dir / "ca.crt")
        cert = str(certs_dir / f"{node_id}.crt")
        key = str(certs_dir / f"{node_id}.key")
        server_ctxs[node_id] = make_server_ssl_context(ca_file, cert, key)
        raft, ledger = _new_raft(
            node_id,
            tmp_path=tmp_path,
            certs_dir=certs_dir,
            members={node_id: "https://127.0.0.1:1"},
            ledger_members=NODE_IDS,
        )
        servers[node_id], ports[node_id] = _start_tls_server(build_raft_handler(raft), server_ctxs[node_id])
        nodes[node_id] = raft
        ledgers[node_id] = ledger

    urls = {node_id: f"https://127.0.0.1:{ports[node_id]}" for node_id in NODE_IDS}
    for node_id in NODE_IDS:
        nodes[node_id].members = dict(urls)
        nodes[node_id].leader_url = urls[node_id] if node_id == LEADER_ID else urls[LEADER_ID]
        nodes[node_id].leader_id = node_id if node_id == LEADER_ID else LEADER_ID
        if node_id == LEADER_ID:
            nodes[node_id].role = "leader"

    return {
        "nodes": nodes,
        "ledgers": ledgers,
        "servers": servers,
        "urls": urls,
        "server_ctxs": server_ctxs,
    }


def _stop_cluster(cluster: dict) -> None:
    for server in list(cluster["servers"].values()):
        _stop_server(server)
    cluster["servers"].clear()


def _restart_node(cluster: dict, node_id: str, tmp_path, certs_dir) -> None:
    server = cluster["servers"].pop(node_id, None)
    if server is not None:
        _stop_server(server)
    raft, ledger = _new_raft(
        node_id,
        tmp_path=tmp_path,
        certs_dir=certs_dir,
        members=cluster["urls"],
        ledger_members=NODE_IDS,
        leader_url=cluster["urls"][LEADER_ID],
    )
    server, port = _start_tls_server(build_raft_handler(raft), cluster["server_ctxs"][node_id])
    cluster["servers"][node_id] = server
    cluster["nodes"][node_id] = raft
    cluster["ledgers"][node_id] = ledger
    cluster["urls"][node_id] = f"https://127.0.0.1:{port}"
    # Republish the restarted node's address to every member.
    for member in cluster["nodes"].values():
        member.members[node_id] = cluster["urls"][node_id]


def _doc_manifest(doc_id: str, sha: str = HEX_A) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Synthetic {doc_id}",
        "shard_owner": "node1-maestro",
        "metadata_holder": "node2-peer",
        "pdf_sha256": sha,
        "chunks_sha256": sha,
        "num_chunks": 1,
        "page_count": 1,
    }


def _assert_tips_match(cluster: dict, expected_index: int) -> None:
    tips = {
        node_id: (
            cluster["ledgers"][node_id].snapshot()["last_index"],
            cluster["ledgers"][node_id].snapshot()["last_hash"],
        )
        for node_id in NODE_IDS
    }
    hashes = {tip[1] for tip in tips.values()}
    assert all(tip[0] == expected_index for tip in tips.values()), tips
    assert len(hashes) == 1, tips


def test_three_node_cluster_commits_catches_up_and_needs_two(tmp_path, certs):
    cluster = _build_three_node_cluster(tmp_path, certs)
    leader = cluster["nodes"][LEADER_ID]
    try:
        first = leader.propose("DOC_MANIFEST", _doc_manifest("doc-1", HEX_A))
        assert first["index"] == 1
        _assert_tips_match(cluster, 1)

        # One follower down: the remaining two are still a majority of three.
        _stop_server(cluster["servers"].pop("node3-peer"))
        second = leader.propose("DOC_MANIFEST", _doc_manifest("doc-2", HEX_B))
        assert second["index"] == 2
        for node_id in ("node1-maestro", "node2-peer"):
            snap = cluster["ledgers"][node_id].snapshot()
            assert snap["last_index"] == 2
            assert snap["last_hash"] == second["entry_hash"]

        # Restarted follower catches up from the leader and reaches the same tip hash.
        _restart_node(cluster, "node3-peer", tmp_path, certs)
        leader.replicate_log()
        leader_snap = cluster["ledgers"][LEADER_ID].snapshot()
        restarted = cluster["ledgers"]["node3-peer"].snapshot()
        assert restarted["last_index"] == leader_snap["last_index"] == 2
        assert restarted["last_hash"] == leader_snap["last_hash"] == second["entry_hash"]

        # Two followers down: one of three cannot commit and the leader's log is untouched.
        _stop_server(cluster["servers"].pop("node2-peer"))
        _stop_server(cluster["servers"].pop("node3-peer"))
        height_before = cluster["ledgers"][LEADER_ID].snapshot()["height"]
        with pytest.raises(ValueError, match="quorum not reached"):
            leader.propose("DOC_MANIFEST", _doc_manifest("doc-3", HEX_C))
        assert cluster["ledgers"][LEADER_ID].snapshot()["height"] == height_before
    finally:
        _stop_cluster(cluster)


def test_three_member_election_requires_a_majority(tmp_path, certs):
    cluster = _build_three_node_cluster(tmp_path, certs)
    candidate = cluster["nodes"]["node2-peer"]
    try:
        _stop_server(cluster["servers"].pop("node3-peer"))
        won = candidate.start_election()
        assert won["elected"] is True
        assert won["votes"] == ["node1-maestro", "node2-peer"]
        assert won["quorum"] == 2
        assert candidate.role == "leader"

        _stop_server(cluster["servers"].pop(LEADER_ID))
        lost = candidate.start_election()
        assert lost["elected"] is False
        assert lost["votes"] == ["node2-peer"]
        assert lost["quorum"] == 2
        assert candidate.role == "follower"
    finally:
        _stop_cluster(cluster)


def _lora_finalize(round_id: int, participants: list[str], hashes: dict[str, str]) -> dict:
    return {
        "round_id": round_id,
        "global_adapter_sha256": HEX_D,
        "participants": participants,
        "participant_hashes": hashes,
        "total_examples": 10,
        "weighted_loss": 0.5,
    }


def _commit(ledger: LedgerState, tx_type: str, payload: dict) -> dict:
    entry = ledger.validate_and_build_entry(1, tx_type, payload)
    ledger.apply_entry(entry)
    return entry


def _submit(ledger: LedgerState, round_id: int, node_id: str, sha: str) -> None:
    _commit(
        ledger,
        "LORA_CLIENT_SUBMIT",
        {"round_id": round_id, "node_id": node_id, "adapter_sha256": sha, "num_examples": 5},
    )


def test_lora_finalize_quorum_is_majority_of_configured_members(tmp_path):
    assert majority(2) == 2
    assert majority(3) == 2
    assert majority(5) == 3

    ledger = LedgerState(tmp_path / "ledger.json", members=NODE_IDS)
    assert ledger.lora_quorum == 2

    for node_id, sha in zip(NODE_IDS, (HEX_A, HEX_B, HEX_C), strict=True):
        _submit(ledger, 1, node_id, sha)
    _commit(
        ledger,
        "LORA_ROUND_FINALIZE",
        _lora_finalize(1, ["node1-maestro", "node2-peer"], {"node1-maestro": HEX_A, "node2-peer": HEX_B}),
    )


def test_lora_finalize_rejects_one_of_three(tmp_path):
    ledger = LedgerState(tmp_path / "ledger.json", members=NODE_IDS)
    _submit(ledger, 2, "node1-maestro", HEX_A)

    with pytest.raises(ValueError, match="requires submissions from 2"):
        ledger.validate(
            "LORA_ROUND_FINALIZE",
            _lora_finalize(2, ["node1-maestro"], {"node1-maestro": HEX_A}),
        )


def test_lora_finalize_rejects_non_member_and_duplicate_submitters(tmp_path):
    ledger = LedgerState(tmp_path / "ledger.json", members=NODE_IDS)
    with pytest.raises(ValueError, match="non-member"):
        ledger.validate(
            "LORA_CLIENT_SUBMIT",
            {"round_id": 3, "node_id": "node4-outsider", "adapter_sha256": HEX_A, "num_examples": 5},
        )

    _submit(ledger, 3, "node1-maestro", HEX_A)
    with pytest.raises(ValueError, match="Non-member submitter"):
        ledger.validate(
            "LORA_ROUND_FINALIZE",
            _lora_finalize(
                3,
                ["node1-maestro", "node4-outsider"],
                {"node1-maestro": HEX_A, "node4-outsider": HEX_B},
            ),
        )

    with pytest.raises(ValueError, match="Duplicate submitter"):
        ledger.validate(
            "LORA_ROUND_FINALIZE",
            _lora_finalize(3, ["node1-maestro", "node1-maestro"], {"node1-maestro": HEX_A}),
        )


def test_duplicate_conflicting_client_submit_is_rejected(tmp_path):
    ledger = LedgerState(tmp_path / "ledger.json", members=NODE_IDS)
    _submit(ledger, 4, "node1-maestro", HEX_A)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        ledger.validate(
            "LORA_CLIENT_SUBMIT",
            {"round_id": 4, "node_id": "node1-maestro", "adapter_sha256": HEX_B, "num_examples": 5},
        )


def test_configured_member_urls_drive_membership():
    members = parse_member_urls(
        "node1-maestro=https://node1-maestro:7050,"
        "node2-peer=https://node2-peer:7050,"
        "node3-peer=https://node3-peer:7050"
    )
    assert members == {
        "node1-maestro": "https://node1-maestro:7050",
        "node2-peer": "https://node2-peer:7050",
        "node3-peer": "https://node3-peer:7050",
    }
    assert majority(len(members)) == 2

    with pytest.raises(ValueError, match="must be node_id=url"):
        parse_member_urls("node1-maestro")
    with pytest.raises(ValueError, match="duplicate raft member"):
        parse_member_urls("node1-maestro=https://a:7050,node1-maestro=https://b:7050")
