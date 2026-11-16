"""Raft Consensus + Hash-Chained Ledger with TLS 1.3 mTLS & Chaincode Validation.

Implements the Hyperledger Fabric 2.x Raft Ordering & Validation pattern in Python:
- Leader (`node1-maestro`) validates proposed transactions, appends a hash-chained block entry,
  replicates via `POST /raft/append` over mTLS (`ssl.CERT_REQUIRED`) to followers (`node2-peer`),
  and commits once quorum acknowledges.
- Followers forward any `/tx/propose` call over mTLS to the Leader and apply replicated entries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GENESIS_ZEROS = "0" * 64
GENESIS_TIMESTAMP = "2026-01-01T00:00:00Z"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_entry_hash(entry: dict) -> str:
    payload_str = canonical_json(entry["payload"])
    raw = (
        f"{entry['index']}|{entry['term']}|{entry['tx_type']}|"
        f"{entry['prev_hash']}|{payload_str}|{entry['timestamp']}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class LedgerState:
    """Hash-chained block log and state machine for sharded RAG & Federated LoRA."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._lock = threading.RLock()
        self.storage_path = Path(storage_path) if storage_path else None
        self.entries: list[dict] = []
        self.documents: dict[str, dict] = {}
        self.backups: dict[str, list[dict]] = {}
        self.client_submissions: dict[str, dict[str, dict]] = {}
        self.final_rounds: dict[str, dict] = {}

        if self.storage_path and self.storage_path.is_file():
            data = json.loads(self.storage_path.read_text())
            self.entries = data["entries"]
            self.documents = data["documents"]
            self.backups = data["backups"]
            self.client_submissions = data["client_submissions"]
            self.final_rounds = data["final_rounds"]
            return

        genesis = {
            "index": 0,
            "term": 1,
            "tx_type": "GENESIS",
            "payload": {
                "network": "fabric-fed-rag-raft-v2",
                "note": "Genesis block for 50/50 sharded RAG & Federated LoRA",
            },
            "prev_hash": GENESIS_ZEROS,
            "timestamp": GENESIS_TIMESTAMP,
        }
        genesis["entry_hash"] = compute_entry_hash(genesis)
        self.entries.append(genesis)
        self._persist()

    def validate(self, tx_type: str, payload: dict) -> None:
        if tx_type == "DOC_MANIFEST":
            doc_id = payload.get("doc_id", "")
            pdf_sha = payload.get("pdf_sha256", "")
            chunks_sha = payload.get("chunks_sha256", "")
            if not doc_id or len(pdf_sha) != 64 or len(chunks_sha) != 64:
                raise ValueError("DOC_MANIFEST requires doc_id and 64-char sha256 hashes")
            if not payload.get("shard_owner") or not payload.get("metadata_holder"):
                raise ValueError("DOC_MANIFEST requires shard_owner and metadata_holder")
            return

        if tx_type == "BACKUP_CHECK":
            doc_id = payload.get("doc_id", "")
            doc = self.documents.get(doc_id)
            if not doc:
                raise ValueError(f"Unknown doc_id {doc_id!r} for BACKUP_CHECK")
            computed = payload.get("computed_sha256", "")
            if computed != doc["pdf_sha256"]:
                raise ValueError(
                    f"Backup checksum mismatch for {doc_id}: got {computed}, ledger expects {doc['pdf_sha256']}"
                )
            if payload.get("ipfs_cid") and doc.get("ipfs_cid") and payload["ipfs_cid"] != doc["ipfs_cid"]:
                raise ValueError(
                    f"Backup IPFS CID mismatch for {doc_id}: got {payload['ipfs_cid']}, ledger expects {doc['ipfs_cid']}"
                )
            return

        if tx_type == "LORA_CLIENT_SUBMIT":
            round_id = int(payload.get("round_id", 0))
            node_id = payload.get("node_id", "")
            sha = payload.get("adapter_sha256", "")
            num_ex = int(payload.get("num_examples", 0))
            if round_id <= 0 or not node_id or len(sha) != 64 or num_ex <= 0:
                raise ValueError("Invalid LORA_CLIENT_SUBMIT fields")
            return

        if tx_type == "LORA_ROUND_FINALIZE":
            round_id = int(payload.get("round_id", 0))
            global_sha = payload.get("global_adapter_sha256", "")
            part_hashes = payload.get("participant_hashes") or {}
            if round_id <= 0 or len(global_sha) != 64 or not part_hashes:
                raise ValueError("LORA_ROUND_FINALIZE requires round_id, global_adapter_sha256, participant_hashes")
            subs = self.client_submissions.get(str(round_id), {})
            for node_id, expected_hash in part_hashes.items():
                recorded = subs.get(node_id)
                if not recorded:
                    raise ValueError(f"Cannot finalize round {round_id}: missing LORA_CLIENT_SUBMIT for {node_id}")
                if recorded["adapter_sha256"] != expected_hash:
                    raise ValueError(f"Adapter hash mismatch for {node_id} in round {round_id}")
            return

        raise ValueError(f"Unsupported tx_type: {tx_type!r}")

    def validate_and_build_entry(self, term: int, tx_type: str, payload: dict) -> dict:
        with self._lock:
            self.validate(tx_type, payload)
            last = self.entries[-1]
            entry = {
                "index": last["index"] + 1,
                "term": int(term),
                "tx_type": tx_type,
                "payload": payload,
                "prev_hash": last["entry_hash"],
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            entry["entry_hash"] = compute_entry_hash(entry)
            return entry

    def apply_entry(self, entry: dict) -> None:
        with self._lock:
            idx = int(entry["index"])
            if idx < len(self.entries):
                if self.entries[idx]["entry_hash"] == entry["entry_hash"]:
                    return
                raise ValueError(f"Conflicting entry at index {idx}")
            if idx != len(self.entries):
                raise ValueError(f"Out-of-order entry index {idx} (expected {len(self.entries)})")

            prev = self.entries[-1]
            if entry["prev_hash"] != prev["entry_hash"]:
                raise ValueError(f"Broken hash chain at index {idx}")
            if compute_entry_hash(entry) != entry["entry_hash"]:
                raise ValueError(f"Entry hash verification failed at index {idx}")

            tx_type = entry["tx_type"]
            payload = entry["payload"]
            self.validate(tx_type, payload)

            if tx_type == "DOC_MANIFEST":
                self.documents[payload["doc_id"]] = payload
            elif tx_type == "BACKUP_CHECK":
                att = {**payload, "verified": True}
                self.backups.setdefault(payload["doc_id"], []).append(att)
            elif tx_type == "LORA_CLIENT_SUBMIT":
                r_key = str(payload["round_id"])
                self.client_submissions.setdefault(r_key, {})[payload["node_id"]] = payload
            elif tx_type == "LORA_ROUND_FINALIZE":
                r_key = str(payload["round_id"])
                self.final_rounds[r_key] = payload

            self.entries.append(entry)
            self._persist()

    def snapshot(self) -> dict:
        with self._lock:
            last = self.entries[-1]
            return {
                "height": len(self.entries),
                "last_index": last["index"],
                "last_hash": last["entry_hash"],
                "documents": self.documents,
                "backups": self.backups,
                "client_submissions": self.client_submissions,
                "final_rounds": self.final_rounds,
                "entries": self.entries,
            }

    def _persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(
                {
                    "entries": self.entries,
                    "documents": self.documents,
                    "backups": self.backups,
                    "client_submissions": self.client_submissions,
                    "final_rounds": self.final_rounds,
                },
                indent=2,
            )
        )
        self.storage_path.chmod(0o600)


class RaftNode:
    """Raft Leader/Follower node replicating log entries over TLS 1.3 mTLS."""

    def __init__(
        self,
        *,
        node_id: str,
        role: str,
        leader_url: str,
        peer_urls: list[str],
        ssl_client_ctx: ssl.SSLContext,
        ledger: LedgerState,
    ) -> None:
        self._lock = threading.Lock()
        self.node_id = node_id
        self.role = role
        self.current_term = 1
        self.leader_id = node_id if role == "leader" else ""
        self.leader_url = leader_url.rstrip("/")
        self.peer_urls = [u.rstrip("/") for u in peer_urls if u.strip()]
        self.ssl_client_ctx = ssl_client_ctx
        self.ledger = ledger
        self.peer_status: dict[str, str] = {}

    def _post_json(self, url: str, payload: dict) -> dict:
        raw = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=raw, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, context=self.ssl_client_ctx, timeout=10.0) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise ValueError(f"HTTP {exc.code} from {url}: {body}") from exc

    def propose(self, tx_type: str, payload: dict) -> dict:
        if self.role != "leader":
            return self._post_json(
                f"{self.leader_url}/tx/propose",
                {"tx_type": tx_type, "payload": payload},
            )

        with self._lock:
            entry = self.ledger.validate_and_build_entry(self.current_term, tx_type, payload)
            snap = self.ledger.snapshot()
            all_entries = [*snap["entries"][1:], entry]
            append_req = {
                "term": self.current_term,
                "leader_id": self.node_id,
                "prev_log_index": 0,
                "prev_log_hash": snap["entries"][0]["entry_hash"],
                "entries": all_entries,
                "leader_commit": entry["index"],
            }

            acks = 1
            total_nodes = 1 + len(self.peer_urls)
            quorum = (total_nodes // 2) + 1

            for peer_url in self.peer_urls:
                try:
                    resp = self._post_json(f"{peer_url}/raft/append", append_req)
                    if resp.get("success"):
                        acks += 1
                        self.peer_status[peer_url] = f"synced (index={resp.get('last_index')}, mTLS=true)"
                    else:
                        self.peer_status[peer_url] = f"rejected: {resp.get('error')}"
                except Exception as exc:
                    self.peer_status[peer_url] = f"replication error: {exc}"

            if acks < quorum:
                raise ValueError(
                    f"raft quorum not reached: got {acks}/{total_nodes} acks (need {quorum}); "
                    f"peers={self.peer_status}"
                )

            self.ledger.apply_entry(entry)
            return entry

    def handle_append_entries(self, req: dict) -> dict:
        with self._lock:
            term = int(req.get("term", 1))
            if term < self.current_term:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "error": "stale term",
                }
            self.current_term = term
            self.leader_id = req.get("leader_id", "")

            try:
                for entry in req.get("entries") or []:
                    self.ledger.apply_entry(entry)
            except Exception as exc:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "error": str(exc),
                }

            snap = self.ledger.snapshot()
            return {
                "node_id": self.node_id,
                "term": self.current_term,
                "success": True,
                "last_index": snap["last_index"],
            }

    def status(self) -> dict:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "term": self.current_term,
            "leader_id": self.leader_id,
            "peer_status": dict(self.peer_status),
            "ledger": self.ledger.snapshot(),
        }


def build_raft_handler(raft_node: RaftNode):
    class RaftHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode())

        def do_GET(self) -> None:
            if self.path in ("/health", "/ledger"):
                cert = self.connection.getpeercert()
                cn = "mTLS-verified"
                if cert and "subject" in cert:
                    for item in cert["subject"]:
                        for k, v in item:
                            if k == "commonName":
                                cn = v
                snap = raft_node.status()
                snap["verified_client_cn"] = cn
                self._send(200, snap)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/raft/append":
                    self._send(200, raft_node.handle_append_entries(body))
                elif self.path == "/tx/propose":
                    entry = raft_node.propose(body["tx_type"], body["payload"])
                    self._send(200, entry)
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

    return RaftHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="FabricFedRAG Python Raft + mTLS Node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--role", default="leader", choices=("leader", "follower"))
    parser.add_argument("--listen-port", type=int, default=7050)
    parser.add_argument("--leader-url", required=True)
    parser.add_argument("--peers", default="")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--ledger", default="./data/ledger.json")
    args = parser.parse_args()

    client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=args.ca)
    client_ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    client_ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)

    server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    server_ctx.verify_mode = ssl.CERT_REQUIRED
    server_ctx.load_verify_locations(cafile=args.ca)
    server_ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)

    peers = [p.strip() for p in args.peers.split(",") if p.strip()]
    ledger = LedgerState(args.ledger)
    node = RaftNode(
        node_id=args.node_id,
        role=args.role,
        leader_url=args.leader_url,
        peer_urls=peers,
        ssl_client_ctx=client_ctx,
        ledger=ledger,
    )

    def heartbeat_loop() -> None:
        while True:
            time.sleep(2.0)
            if node.role == "leader" and node.peer_urls:
                snap = node.ledger.snapshot()
                req = {
                    "term": node.current_term,
                    "leader_id": node.node_id,
                    "prev_log_index": 0,
                    "prev_log_hash": snap["entries"][0]["entry_hash"],
                    "entries": snap["entries"][1:],
                    "leader_commit": snap["last_index"],
                }
                for peer_url in node.peer_urls:
                    try:
                        resp = node._post_json(f"{peer_url}/raft/append", req)
                        if resp.get("success"):
                            node.peer_status[peer_url] = f"synced (index={resp.get('last_index')}, mTLS=true)"
                    except Exception as exc:
                        node.peer_status[peer_url] = f"unreachable: {exc}"

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.listen_port), build_raft_handler(node))
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    print(f"[raft_ledger] {args.node_id} ({args.role}) listening with mTLS on :{args.listen_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
