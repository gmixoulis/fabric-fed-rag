"""Raft Consensus + Hash-Chained Ledger with TLS 1.3 mTLS & Chaincode Validation.

Implements the Hyperledger Fabric 2.x Raft Ordering & Validation pattern in Python:
- The configured Leader validates proposed transactions, appends a hash-chained block entry,
  replicates via `POST /raft/append` over mTLS (`ssl.CERT_REQUIRED`) to every other configured
  member, and commits once a majority of the membership acknowledges.
- Followers forward any `/tx/propose` call over mTLS to the Leader and apply replicated entries.
- Membership is supplied by configuration (`--member-urls` or the `RaftNode(members=...)`
  argument). An election also needs a majority of that membership.
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


def majority(node_count: int) -> int:
    """Return the smallest majority for a membership of ``node_count`` nodes."""
    if node_count <= 0:
        raise ValueError("a raft membership needs at least one node")
    return node_count // 2 + 1


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_entry_hash(entry: dict) -> str:
    payload_str = canonical_json(entry["payload"])
    raw = (
        f"{entry['index']}|{entry['term']}|{entry['tx_type']}|"
        f"{entry['prev_hash']}|{payload_str}|{entry['timestamp']}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


class LedgerState:
    """Hash-chained block log and state machine for sharded RAG & Federated LoRA."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        *,
        members: list[str] | None = None,
        lora_quorum: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.storage_path = Path(storage_path) if storage_path else None
        self.members = [m for m in (members or []) if m]
        if lora_quorum is None:
            # Without configured membership (unit tests, legacy two-node callers) one
            # valid submission is the historical behaviour. With membership the
            # default is the majority of members.
            self.lora_quorum = majority(len(self.members)) if self.members else 1
        else:
            self.lora_quorum = int(lora_quorum)
            if self.lora_quorum <= 0:
                raise ValueError("lora_quorum must be positive")
            if self.members and self.lora_quorum > len(self.members):
                raise ValueError("lora_quorum cannot exceed the number of configured members")
        self.entries: list[dict] = []
        self.documents: dict[str, dict] = {}
        self.backups: dict[str, list[dict]] = {}
        self.client_submissions: dict[str, dict[str, dict]] = {}
        self.final_rounds: dict[str, dict] = {}

        if self.storage_path and self.storage_path.is_file():
            data = json.loads(self.storage_path.read_text())
            self._load_entries(data["entries"])
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
            if not doc_id or not _is_sha256_hex(pdf_sha) or not _is_sha256_hex(chunks_sha):
                raise ValueError("DOC_MANIFEST requires doc_id and 64-char sha256 hashes")
            if not payload.get("shard_owner") or not payload.get("metadata_holder"):
                raise ValueError("DOC_MANIFEST requires shard_owner and metadata_holder")
            existing = self.documents.get(doc_id)
            if existing is not None and existing != payload:
                raise ValueError(f"DOC_MANIFEST already recorded for {doc_id!r} with a different payload")
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
            if round_id <= 0 or not node_id or not _is_sha256_hex(sha) or num_ex <= 0:
                raise ValueError("Invalid LORA_CLIENT_SUBMIT fields")
            if self.members and node_id not in self.members:
                raise ValueError(f"LORA_CLIENT_SUBMIT from non-member {node_id!r}")
            recorded = self.client_submissions.get(str(round_id), {}).get(node_id)
            if recorded is not None and recorded.get("adapter_sha256") != sha:
                raise ValueError(f"Conflicting duplicate LORA_CLIENT_SUBMIT for {node_id} in round {round_id}")
            return

        if tx_type == "LORA_ROUND_FINALIZE":
            round_id = int(payload.get("round_id", 0))
            global_sha = payload.get("global_adapter_sha256", "")
            part_hashes = payload.get("participant_hashes") or {}
            if round_id <= 0 or not _is_sha256_hex(global_sha) or not part_hashes:
                raise ValueError("LORA_ROUND_FINALIZE requires round_id, global_adapter_sha256, participant_hashes")
            if not isinstance(part_hashes, dict):
                raise ValueError("LORA_ROUND_FINALIZE participant_hashes must be a mapping")

            participants = payload.get("participants")
            if participants is not None:
                if not isinstance(participants, list) or not all(isinstance(p, str) and p for p in participants):
                    raise ValueError("LORA_ROUND_FINALIZE participants must be a list of node ids")
                if len(participants) != len(set(participants)):
                    raise ValueError(f"Duplicate submitter in LORA_ROUND_FINALIZE for round {round_id}")
                if set(participants) != set(part_hashes):
                    raise ValueError(
                        f"LORA_ROUND_FINALIZE participants do not match participant_hashes for round {round_id}"
                    )

            for node_id, expected_hash in part_hashes.items():
                if not _is_sha256_hex(expected_hash):
                    raise ValueError(f"Invalid participant hash for {node_id} in round {round_id}")
                if self.members and node_id not in self.members:
                    raise ValueError(f"Non-member submitter {node_id!r} in LORA_ROUND_FINALIZE for round {round_id}")
            subs = self.client_submissions.get(str(round_id), {})
            for node_id, expected_hash in part_hashes.items():
                recorded = subs.get(node_id)
                if not recorded:
                    raise ValueError(f"Cannot finalize round {round_id}: missing LORA_CLIENT_SUBMIT for {node_id}")
                if recorded["adapter_sha256"] != expected_hash:
                    raise ValueError(f"Adapter hash mismatch for {node_id} in round {round_id}")
            if len(part_hashes) < self.lora_quorum:
                raise ValueError(
                    f"LORA_ROUND_FINALIZE for round {round_id} requires submissions from {self.lora_quorum} "
                    f"member(s), got {len(part_hashes)}"
                )
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

    def _apply_payload_state(self, tx_type: str, payload: dict) -> None:
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

    def _load_entries(self, entries: list[dict]) -> None:
        if not entries:
            raise ValueError("Ledger has no entries")
        for i, entry in enumerate(entries):
            if compute_entry_hash(entry) != entry.get("entry_hash"):
                raise ValueError(f"Entry hash verification failed at index {i}")
            expected_prev = GENESIS_ZEROS if i == 0 else entries[i - 1]["entry_hash"]
            if entry.get("prev_hash") != expected_prev:
                raise ValueError(f"Broken hash chain at index {i}")
        for entry in entries:
            self._apply_payload_state(entry["tx_type"], entry["payload"])
            self.entries.append(entry)

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
            self._apply_payload_state(tx_type, payload)

            self.entries.append(entry)
            self._persist()

    def snapshot(self) -> dict:
        with self._lock:
            last = self.entries[-1]
            return {
                "height": len(self.entries),
                "last_index": last["index"],
                "last_hash": last["entry_hash"],
                "lora_quorum": self.lora_quorum,
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
        members: dict[str, str] | None = None,
        election_timeout: float = 8.0,
    ) -> None:
        self._lock = threading.Lock()
        self.node_id = node_id
        self.role = role
        self.current_term = 1
        self.voted_for = ""
        self.leader_id = node_id if role == "leader" else ""
        self.leader_url = leader_url.rstrip("/")
        self.peer_urls = [u.rstrip("/") for u in peer_urls if u.strip()]
        self.election_timeout = float(election_timeout)
        self.last_leader_contact = time.monotonic()
        # ``members`` maps every raft member id to its base URL and must include this
        # node. Without it the legacy peer-URL list defines the membership.
        self._membership_configured = bool(members)
        configured = {nid.strip(): url.rstrip("/") for nid, url in (members or {}).items() if nid and url}
        if configured:
            configured.setdefault(node_id, self.leader_url)
        self.members = configured
        self.ssl_client_ctx = ssl_client_ctx
        self.ledger = ledger
        self.peer_status: dict[str, str] = {}

    @property
    def total_nodes(self) -> int:
        return len(self.members) if self._membership_configured else 1 + len(self.peer_urls)

    @property
    def quorum(self) -> int:
        return majority(self.total_nodes)

    def peer_targets(self) -> dict[str, str]:
        """Return member id -> URL for every other voting member."""
        if self._membership_configured:
            return {nid: url for nid, url in self.members.items() if nid != self.node_id}
        return {url: url for url in self.peer_urls}

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
                "leader_url": self.leader_url,
                "prev_log_index": 0,
                "prev_log_hash": snap["entries"][0]["entry_hash"],
                "entries": all_entries,
                "leader_commit": entry["index"],
            }

            acks = 1
            quorum = self.quorum
            for peer_url in self.peer_targets().values():
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
                    f"raft quorum not reached: got {acks}/{self.total_nodes} acks (need {quorum}); "
                    f"peers={self.peer_status}"
                )

            self.ledger.apply_entry(entry)
            return entry

    def replicate_log(self) -> dict[str, str]:
        """Leader heartbeat: push the full log so restarted or lagging members catch up."""
        if self.role != "leader":
            return {}
        snap = self.ledger.snapshot()
        req = {
            "term": self.current_term,
            "leader_id": self.node_id,
            "leader_url": self.leader_url,
            "prev_log_index": 0,
            "prev_log_hash": snap["entries"][0]["entry_hash"],
            "entries": snap["entries"][1:],
            "leader_commit": snap["last_index"],
        }
        for peer_url in self.peer_targets().values():
            try:
                resp = self._post_json(f"{peer_url}/raft/append", req)
                if resp.get("success"):
                    self.peer_status[peer_url] = f"synced (index={resp.get('last_index')}, mTLS=true)"
                else:
                    self.peer_status[peer_url] = f"rejected: {resp.get('error')}"
            except Exception as exc:
                self.peer_status[peer_url] = f"unreachable: {exc}"
        return dict(self.peer_status)

    def handle_append_entries(self, req: dict) -> dict:
        with self._lock:
            term = int(req.get("term", 1))
            leader_id = req.get("leader_id", "")
            if self._membership_configured and leader_id not in self.members:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "error": f"unknown leader {leader_id!r}",
                }
            if term < self.current_term:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "error": "stale term",
                }
            if term > self.current_term:
                self.current_term = term
                self.voted_for = ""
                if self.role == "leader":
                    self.role = "follower"
            self.leader_id = leader_id
            if leader_id in self.members:
                self.leader_url = self.members[leader_id]
            self.last_leader_contact = time.monotonic()

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

    def handle_request_vote(self, req: dict) -> dict:
        with self._lock:
            term = int(req.get("term", 0))
            candidate_id = req.get("candidate_id", "")
            if self._membership_configured and candidate_id not in self.members:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "granted": False,
                    "error": f"unknown candidate {candidate_id!r}",
                }
            if term < self.current_term:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "granted": False,
                    "error": "stale term",
                }
            local = self.ledger.snapshot()
            last_index = int(req.get("last_log_index", -1))
            last_hash = req.get("last_log_hash", "")
            up_to_date = last_index > local["last_index"] or (
                last_index == local["last_index"] and (not last_hash or last_hash == local["last_hash"])
            )
            if not up_to_date:
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "granted": False,
                    "error": "candidate log is behind",
                }
            if term > self.current_term:
                self.current_term = term
                self.voted_for = ""
                if self.role == "leader":
                    self.role = "follower"
            if self.voted_for not in ("", candidate_id):
                return {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "granted": False,
                    "error": f"already voted for {self.voted_for!r}",
                }
            self.voted_for = candidate_id
            self.leader_id = ""
            return {"node_id": self.node_id, "term": self.current_term, "granted": True}

    def start_election(self) -> dict:
        """Campaign for leadership; wins only with votes from a majority of members."""
        if not self._membership_configured:
            raise ValueError("leader election requires an explicit member configuration")
        self.current_term += 1
        term = self.current_term
        self.role = "candidate"
        self.voted_for = self.node_id
        self.leader_id = ""
        snap = self.ledger.snapshot()
        votes = {self.node_id}
        errors: dict[str, str] = {}
        for candidate_id, url in self.peer_targets().items():
            try:
                resp = self._post_json(
                    f"{url}/raft/vote",
                    {
                        "term": term,
                        "candidate_id": self.node_id,
                        "last_log_index": snap["last_index"],
                        "last_log_hash": snap["last_hash"],
                    },
                )
                if resp.get("granted") and int(resp.get("term", term)) == term:
                    votes.add(candidate_id)
                else:
                    errors[candidate_id] = str(resp.get("error", "vote denied"))
            except Exception as exc:
                errors[candidate_id] = str(exc)
        elected = len(votes) >= self.quorum
        if elected:
            self.role = "leader"
            self.leader_id = self.node_id
            self.leader_url = self.members.get(self.node_id, self.leader_url)
            self.replicate_log()
        else:
            self.role = "follower"
            self.last_leader_contact = time.monotonic()
        return {
            "node_id": self.node_id,
            "elected": elected,
            "term": term,
            "votes": sorted(votes),
            "quorum": self.quorum,
            "errors": errors,
        }

    def status(self) -> dict:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "term": self.current_term,
            "leader_id": self.leader_id,
            "members": sorted(self.members) if self._membership_configured else [],
            "quorum": self.quorum,
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
                elif self.path == "/raft/vote":
                    self._send(200, raft_node.handle_request_vote(body))
                elif self.path == "/tx/propose":
                    entry = raft_node.propose(body["tx_type"], body["payload"])
                    self._send(200, entry)
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

    return RaftHandler


def parse_member_urls(raw: str) -> dict[str, str]:
    """Parse ``id=url,id=url`` into a membership map."""
    members: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"member entry {item!r} must be node_id=url")
        node_id, url = item.split("=", 1)
        node_id, url = node_id.strip(), url.strip().rstrip("/")
        if not node_id or not url:
            raise ValueError(f"member entry {item!r} must be node_id=url")
        if node_id in members:
            raise ValueError(f"duplicate raft member {node_id!r}")
        members[node_id] = url
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description="FabricFedRAG Python Raft + mTLS Node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--role", default="leader", choices=("leader", "follower"))
    parser.add_argument("--listen-port", type=int, default=7050)
    parser.add_argument("--leader-url", required=True)
    parser.add_argument("--peers", default="", help="Legacy peer URL list (membership of 1 + peers)")
    parser.add_argument(
        "--member-urls",
        default="",
        help="Full raft membership as node_id=url pairs, including this node",
    )
    parser.add_argument("--lora-quorum", type=int, default=None, help="LORA finalize quorum (default: majority)")
    parser.add_argument("--election-timeout", type=float, default=8.0)
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
    members = parse_member_urls(args.member_urls)
    if members and args.node_id not in members:
        members[args.node_id] = args.leader_url.rstrip("/")
    ledger = LedgerState(args.ledger, members=list(members) or None, lora_quorum=args.lora_quorum)
    node = RaftNode(
        node_id=args.node_id,
        role=args.role,
        leader_url=args.leader_url,
        peer_urls=peers,
        ssl_client_ctx=client_ctx,
        ledger=ledger,
        members=members or None,
        election_timeout=args.election_timeout,
    )

    def heartbeat_loop() -> None:
        while True:
            time.sleep(2.0)
            if node.role == "leader":
                node.replicate_log()
            elif node._membership_configured and (
                time.monotonic() - node.last_leader_contact > node.election_timeout
            ):
                result = node.start_election()
                print(f"[raft_ledger] election {result}", flush=True)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.listen_port), build_raft_handler(node))
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    membership = f", members={sorted(members)} quorum={node.quorum}" if members else ""
    print(
        f"[raft_ledger] {args.node_id} ({args.role}) listening with mTLS on :{args.listen_port}{membership}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
