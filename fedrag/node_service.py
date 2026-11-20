"""Per-Node mTLS Service for Sharded RAG & Federated LoRA Training.

Runs on each Docker peer (`node1-maestro`, `node2-peer`, `node3-peer`), enforcing TLS 1.3
mutual authentication (`ssl.CERT_REQUIRED`) for all peer-to-peer and Maestro-to-peer calls,
and committing all document hashes, backup verifications, and LoRA round hashes to the
Python Raft ledger (`fedrag/raft_ledger.py`).

The member list (`--members node_id=url,...`) is the single source of truth this service
uses for three things: the shard ring (`fedrag/shard_store.ShardRing`), the peers it
replicates metadata stubs and one backup replica to, and the peers a scatter-gather query
or a federated round fans out to. A peer that does not answer lowers recall or the number
of training participants; it does not fail the request. The ledger membership and finalize
quorum are configured on the Raft process (`--member-urls`, `--lora-quorum`).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from fedrag.embed import HashEmbedder, OllamaEmbedder
from fedrag.fedavg import (
    adapter_sha256,
    aggregate_adapters,
    load_adapter_npz,
    save_adapter_npz,
    server_lr_for,
)
from fedrag.ipfs_client import KuboIPFSClient, LocalCIDStore
from fedrag.raft_ledger import parse_member_urls
from fedrag.shard_store import ShardedRAGStore
from fedrag.trainer import init_lora_state, train_local_lora_round


def make_client_ssl_context(ca_file: str, cert_file: str, key_file: str) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ctx


def make_server_ssl_context(ca_file: str, cert_file: str, key_file: str) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cafile=ca_file)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ctx


def mtls_request(
    ssl_ctx: ssl.SSLContext,
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"mTLS HTTP {exc.code} from {url}: {body}") from exc


def serialize_state_b64(state: dict[str, np.ndarray]) -> str:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v, dtype=np.float32) for k, v in state.items()})
    return base64.b64encode(buf.getvalue()).decode("ascii")


def deserialize_state_b64(b64_str: str) -> dict[str, np.ndarray]:
    raw = base64.b64decode(b64_str.encode("ascii"))
    with np.load(io.BytesIO(raw)) as data:
        return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}


class NodeApp:
    def __init__(
        self,
        *,
        node_id: str,
        data_dir: Path,
        raft_url: str,
        members: dict[str, str] | None,
        ca_file: str,
        cert_file: str,
        key_file: str,
        embedder,
        ipfs_client=None,
        peer_timeout: float | None = None,
        train_timeout: float | None = None,
    ) -> None:
        self.node_id = node_id
        self.data_dir = Path(data_dir)
        self.raft_url = raft_url.rstrip("/")
        # ``members`` maps every RAG member id to its base URL and defines the shard ring.
        self.members: dict[str, str] = {
            str(mid).strip(): str(url or "").rstrip("/")
            for mid, url in (members or {}).items()
            if str(mid).strip()
        }
        self.members.setdefault(node_id, "")
        self.peers: dict[str, str] = {
            mid: url for mid, url in self.members.items() if mid != node_id and url
        }
        self.ssl_client_ctx = make_client_ssl_context(ca_file, cert_file, key_file)
        self.store = ShardedRAGStore(
            node_id=node_id,
            root_dir=self.data_dir / "rag",
            embedder=embedder,
            members=list(self.members),
        )
        self.ipfs = ipfs_client if ipfs_client is not None else LocalCIDStore(node_id)
        self.peer_timeout = float(
            peer_timeout if peer_timeout is not None else os.environ.get("FEDRAG_PEER_TIMEOUT", "10")
        )
        self.train_timeout = float(
            train_timeout if train_timeout is not None else os.environ.get("FEDRAG_TRAIN_TIMEOUT", "900")
        )
        self.rounds_dir = self.data_dir / "rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- helpers ---
    @property
    def ring(self):
        return self.store.ring

    @property
    def shard_id(self) -> str:
        return self.store.shard_id

    def configure_member_urls(self, members: dict[str, str]) -> None:
        """Publish the member base URLs (used when ports are only known after binding)."""
        if sorted(members) != sorted(self.members):
            raise ValueError(
                f"member ids must match the shard ring: got {sorted(members)}, "
                f"ring={sorted(self.members)}"
            )
        self.members = {mid: str(url).rstrip("/") for mid, url in members.items()}
        self.peers = {
            mid: url for mid, url in self.members.items() if mid != self.node_id and url
        }

    def propose_raft_tx(self, tx_type: str, payload: dict) -> dict:
        return mtls_request(
            self.ssl_client_ctx,
            f"{self.raft_url}/tx/propose",
            method="POST",
            payload={"tx_type": tx_type, "payload": payload},
        )

    def get_raft_ledger(self) -> dict:
        return mtls_request(self.ssl_client_ctx, f"{self.raft_url}/ledger", method="GET")

    def lora_quorum(self) -> int:
        return int(self.get_raft_ledger().get("ledger", {}).get("lora_quorum") or 1)

    def _call_peer(
        self,
        peer_id: str,
        path: str,
        payload: dict | None = None,
        *,
        method: str = "POST",
        timeout: float | None = None,
    ) -> dict:
        url = self.peers.get(peer_id)
        if not url:
            raise KeyError(f"no RAG URL configured for member {peer_id!r}")
        return mtls_request(
            self.ssl_client_ctx,
            f"{url}{path}",
            method=method,
            payload=payload,
            timeout=self.peer_timeout if timeout is None else timeout,
        )

    def _fan_out(
        self,
        path: str,
        payload: dict | None,
        *,
        method: str = "POST",
        timeout: float | None = None,
    ) -> tuple[dict[str, dict], dict[str, str]]:
        """Ask every configured peer in parallel; return (answers, failures)."""
        responses: dict[str, dict] = {}
        failures: dict[str, str] = {}
        if not self.peers:
            return responses, failures
        with ThreadPoolExecutor(max_workers=len(self.peers)) as pool:
            futures = {
                pool.submit(
                    self._call_peer, peer_id, path, payload, method=method, timeout=timeout
                ): peer_id
                for peer_id in self.peers
            }
            for future in as_completed(futures):
                peer_id = futures[future]
                try:
                    responses[peer_id] = future.result()
                except Exception as exc:  # unreachable, slow or rejected: keep the rest
                    failures[peer_id] = str(exc)
        return responses, failures

    def missing_shards(self, missing_nodes: list[str]) -> list[str]:
        return sorted(
            {
                self.ring.shard_id(self.ring.index_of_node(node_id))
                for node_id in missing_nodes
                if node_id in self.ring.members
            }
        )

    def check_embedding_agreement(self) -> dict:
        """One embedding model digest across every reachable node and the ledger (WP2)."""
        digest = self.store.embedder.digest
        report: dict[str, dict] = {self.node_id: {"digest": digest, "status": "local"}}
        responses, failures = self._fan_out("/embedding-model", None, method="GET")
        for peer_id, payload in responses.items():
            peer_digest = payload.get("digest")
            if peer_digest != digest:
                raise ValueError(
                    f"Embedding model digest differs between nodes: "
                    f"{self.node_id}={digest} {peer_id}={peer_digest}"
                )
            report[peer_id] = {"digest": peer_digest, "status": "match"}
        for peer_id, error in failures.items():
            report[peer_id] = {"status": "unreachable", "error": error}

        ledger_docs = self.get_raft_ledger().get("ledger", {}).get("documents", {})
        mismatched = sorted(
            doc_id
            for doc_id, doc in ledger_docs.items()
            if doc.get("embedding_model_digest") != digest
        )
        if mismatched:
            raise ValueError(
                f"Embedding model digest differs from ledger for {mismatched[:3]} "
                f"({len(mismatched)} document(s), ledger expects a different model)"
            )
        return report

    # ------------------------------------------------------------ ingest path ---
    def ingest_full_and_replicate(self, body: dict) -> dict:
        """Store a document in this node's shard: `shard_owner`, `metadata_holder` and the backup
        target all come from the shard ring, so a body-supplied `metadata_holder` is ignored."""
        pdf_bytes = base64.b64decode(body["pdf_base64"].encode("ascii"))
        doc_id = body["doc_id"]
        assignment = self.ring.assignment(doc_id)
        if assignment["owner"] != self.node_id:
            raise ValueError(
                f"document {doc_id} belongs to {assignment['shard_id']} owned by "
                f"{assignment['owner']}; send this ingest to {assignment['owner']}"
            )
        backup_node = assignment["backup"]
        stub_holders = self.ring.stub_holders(doc_id)

        # 0. One embedding model everywhere (local, peers, ledger) before anything is written
        embedding_agreement = self.check_embedding_agreement()

        # 1. Pin the full PDF to the local Kubo IPFS node (Private Swarm) and get the CID
        ipfs_cid = self.ipfs.add_bytes(pdf_bytes, filename=f"{doc_id}.pdf", pin=True)

        manifest = self.store.ingest_full_document(
            doc_id=doc_id,
            title=body["title"],
            pdf_bytes=pdf_bytes,
            chunks=body.get("chunks"),
            page_count=int(body.get("page_count", 1)),
            ipfs_cid=ipfs_cid,
            training_pairs=body.get("training_pairs", []),
        )
        manifest["embedding_model_digest"] = self.store.embedder.digest

        # 2. Commit the document manifest + IPFS CID + SHA-256 hashes to the Raft ledger
        raft_entry = self.propose_raft_tx("DOC_MANIFEST", manifest)

        # 3. Metadata-only stub to every other member, verified backup replica to the ring successor
        warnings: list[str] = []
        stub_acks: dict[str, dict] = {}
        for peer_id in stub_holders:
            if peer_id not in self.peers:
                warnings.append(f"metadata stub for {peer_id}: no RAG URL configured")
                continue
            try:
                stub_acks[peer_id] = self._call_peer(peer_id, "/shard/ingest-stub", manifest)
            except Exception as exc:
                stub_acks[peer_id] = {"error": str(exc)}
                warnings.append(f"metadata stub to {peer_id}: {exc}")

        backup_ack = None
        if backup_node is None or backup_node == self.node_id:
            warnings.append("no backup replica: the ring has a single member")
        elif backup_node not in self.peers:
            warnings.append(f"backup replica on {backup_node}: no RAG URL configured")
        else:
            try:
                backup_ack = self._call_peer(
                    backup_node,
                    "/shard/store-backup",
                    {
                        "doc_id": manifest["doc_id"],
                        "source_node": self.node_id,
                        "ipfs_cid": ipfs_cid,
                        "pdf_base64": body["pdf_base64"],
                        "expected_sha256": manifest["pdf_sha256"],
                    },
                )
            except Exception as exc:
                warnings.append(f"backup replica on {backup_node}: {exc}")

        return {
            "manifest": manifest,
            "shard_assignment": assignment,
            "raft_block_index": raft_entry["index"],
            "raft_entry_hash": raft_entry["entry_hash"],
            "embedding_agreement": embedding_agreement,
            "peer_metadata_stubs": stub_acks,
            "peer_metadata_stub": stub_acks.get(backup_node),
            "peer_backup_attestation": backup_ack,
            "replica_warnings": warnings,
        }

    def store_backup_and_attest(self, body: dict) -> dict:
        doc_id = body["doc_id"]
        if self.ring.num_shards > 1 and self.ring.backup(doc_id) != self.node_id:
            raise ValueError(
                f"{doc_id} is backed up by {self.ring.backup(doc_id)}, not {self.node_id}; "
                "each document has exactly one replica holder on the ring"
            )
        ipfs_cid = body.get("ipfs_cid", "")
        if ipfs_cid:
            try:
                self.ipfs.pin_add(ipfs_cid)
            except Exception:
                if body.get("pdf_base64"):
                    raw_fallback = base64.b64decode(body["pdf_base64"].encode("ascii"))
                    self.ipfs.add_bytes(raw_fallback, filename=f"{body['doc_id']}.backup.pdf", pin=True)

        if body.get("pdf_base64"):
            pdf_bytes = base64.b64decode(body["pdf_base64"].encode("ascii"))
        elif ipfs_cid:
            pdf_bytes = self.ipfs.cat_bytes(ipfs_cid)
        else:
            raise ValueError("store_backup requires either ipfs_cid or pdf_base64")

        attestation = self.store.store_and_verify_backup(
            doc_id=body["doc_id"],
            source_node=body["source_node"],
            pdf_bytes=pdf_bytes,
            expected_sha256=body["expected_sha256"],
            ipfs_cid=ipfs_cid,
        )
        raft_entry = self.propose_raft_tx("BACKUP_CHECK", attestation)
        return {**attestation, "raft_block_index": raft_entry["index"]}

    def export_backup(self, body: dict) -> dict:
        """Return one verified backup replica so its owner can restore its shard."""
        replica = self.store.read_backup(body["doc_id"])
        return {
            **{k: v for k, v in replica.items() if k != "pdf_bytes"},
            "pdf_base64": base64.b64encode(replica["pdf_bytes"]).decode("ascii"),
        }

    def restore_owned_shard(self, body: dict) -> dict:
        """Owner-side recovery: pull verified replicas from the backup holder and re-index them."""
        backup_node = body.get("backup_node") or self.store.backup_target()
        if not backup_node:
            raise ValueError("no backup holder to restore from")
        if backup_node not in self.peers:
            raise ValueError(f"backup holder {backup_node} has no RAG URL configured")
        # The ledger, not the local store, lists what this node owned: after a total loss
        # the local store is empty.
        doc_ids = body.get("doc_ids") or sorted(
            doc_id
            for doc_id in self.get_raft_ledger().get("ledger", {}).get("documents", {})
            if self.store.owns(doc_id)
        )
        restored = []
        failures: dict[str, str] = {}
        for doc_id in doc_ids:
            try:
                exported = self._call_peer(backup_node, "/shard/export-backup", {"doc_id": doc_id})
                replica = {
                    **exported,
                    "pdf_bytes": base64.b64decode(exported["pdf_base64"].encode("ascii")),
                }
                restored.append(self.store.ingest_restored_backup(replica))
            except Exception as exc:
                failures[doc_id] = str(exc)
        return {
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "backup_node": backup_node,
            "restored": restored,
            "restored_documents": len(restored),
            "failures": failures,
        }

    # ---------------------------------------------------------- training path ---
    def train_local_round(self, body: dict) -> dict:
        round_id = int(body["round_id"])
        lora_mode = str(body.get("lora_mode", "fedavg")).lower()
        round_dir = self.rounds_dir / f"{lora_mode}-round-{round_id}" / self.node_id
        global_dir = None
        if body.get("global_adapter_b64"):
            global_state = deserialize_state_b64(body["global_adapter_b64"])
            global_dir = self.rounds_dir / f"{lora_mode}-round-{round_id}" / "global-init"
            save_adapter_npz(global_dir, global_state, {"round_id": round_id - 1})

        examples = self.store.export_training_shard()
        result = train_local_lora_round(
            node_id=self.node_id,
            shard_id=self.shard_id,
            round_id=round_id,
            examples=examples,
            out_dir=round_dir,
            global_adapter_dir=global_dir,
            config={"lora_mode": lora_mode},
        )
        state, _ = load_adapter_npz(round_dir)
        adapter_bytes = (round_dir / "adapter_model.npz").read_bytes()
        adapter_cid = self.ipfs.add_bytes(
            adapter_bytes, filename=f"{self.node_id}-r{round_id}.npz", pin=True
        )

        submission = {
            "round_id": round_id,
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "lora_mode": lora_mode,
            "adapter_cid": adapter_cid,
            "adapter_sha256": result["adapter_sha256"],
            "num_examples": result["num_examples"],
            "first_loss": result["first_loss"],
            "last_loss": result["last_loss"],
        }
        raft_entry = self.propose_raft_tx("LORA_CLIENT_SUBMIT", submission)
        return {
            **result,
            "adapter_cid": adapter_cid,
            "raft_block_index": raft_entry["index"],
            "adapter_b64": serialize_state_b64(state),
        }

    def coordinate_federated_round(self, body: dict) -> dict:
        """Coordinator: train on every reachable member's shard, verify hashes, aggregate, finalize."""
        round_id = int(body["round_id"])
        lora_mode = str(body.get("lora_mode", "fedavg")).lower()
        aggregation = str(body.get("aggregation", "average")).lower()
        server_lr_for(aggregation, 1)  # reject an unknown mode before any training happens

        global_b64 = None
        previous = None
        if round_id > 1:
            prev_global = self.rounds_dir / f"{lora_mode}-round-{round_id - 1}" / "global-final"
            if prev_global.exists():
                previous, _ = load_adapter_npz(prev_global)
                global_b64 = serialize_state_b64(previous)

        # 1. Train on this node's shard
        local_res = self.train_local_round(
            {"round_id": round_id, "lora_mode": lora_mode, "global_adapter_b64": global_b64}
        )
        local_state = deserialize_state_b64(local_res["adapter_b64"])
        if adapter_sha256(local_state) != local_res["adapter_sha256"]:
            raise RuntimeError("Local LoRA adapter SHA-256 verification failed")

        participants = [self.node_id]
        parts = [(local_state, float(local_res["num_examples"]))]
        clients = [{k: v for k, v in local_res.items() if k != "adapter_b64"}]

        # 2. Train on every other reachable member's shard over mTLS
        responses, failures = self._fan_out(
            "/fed/train-local",
            {"round_id": round_id, "lora_mode": lora_mode, "global_adapter_b64": global_b64},
            timeout=self.train_timeout,
        )
        unavailable: dict[str, str] = dict(failures)

        # 3. Verify that every received LoRA tensor matches its SHA-256 commitment
        for peer_id, peer_res in responses.items():
            peer_state = deserialize_state_b64(peer_res["adapter_b64"])
            if adapter_sha256(peer_state) != peer_res["adapter_sha256"]:
                raise RuntimeError(f"Peer LoRA adapter SHA-256 verification failed for {peer_id}")
            participants.append(peer_id)
            parts.append((peer_state, float(peer_res["num_examples"])))
            clients.append({k: v for k, v in peer_res.items() if k != "adapter_b64"})

        # 4. The ledger quorum (2 of 3) decides whether this round may be finalized at all
        quorum = self.lora_quorum()
        if len(participants) < quorum:
            raise ValueError(
                f"round {round_id}: only {len(participants)} of {self.ring.num_shards} members trained "
                f"({sorted(participants)}); LORA_ROUND_FINALIZE needs {quorum}. "
                f"Unavailable: {sorted(unavailable)}"
            )

        # 5. Plain averaging (server_lr = 1) or summed updates with server_lr = participants
        server_lr = server_lr_for(aggregation, len(participants))
        if server_lr != 1.0 and previous is None:
            # Round 1: every participant starts from the same deterministically seeded adapter.
            previous = init_lora_state({"lora_mode": lora_mode})
        global_state = aggregate_adapters(parts, previous=previous, server_lr=server_lr)

        total_examples = int(sum(int(weight) for _, weight in parts))
        weighted_loss = sum(
            float(client["last_loss"]) * int(client["num_examples"]) for client in clients
        ) / total_examples
        global_dir = self.rounds_dir / f"{lora_mode}-round-{round_id}" / "global-final"
        global_sha = save_adapter_npz(
            global_dir,
            global_state,
            {
                "round_id": round_id,
                "lora_mode": lora_mode,
                "aggregation": aggregation,
                "server_lr": server_lr,
                "participants": participants,
                "total_examples": total_examples,
                "weighted_loss": round(weighted_loss, 6),
            },
        )
        global_bytes = (global_dir / "adapter_model.npz").read_bytes()
        global_cid = self.ipfs.add_bytes(
            global_bytes, filename=f"global-{lora_mode}-r{round_id}.npz", pin=True
        )

        # 6. Commit LORA_ROUND_FINALIZE to the Raft ledger (chaincode verifies client hashes)
        finalize_payload = {
            "round_id": round_id,
            "lora_mode": lora_mode,
            "aggregation": aggregation,
            "server_lr": server_lr,
            "global_adapter_cid": global_cid,
            "global_adapter_sha256": global_sha,
            "participants": participants,
            "participant_hashes": {client["node_id"]: client["adapter_sha256"] for client in clients},
            "unavailable_members": sorted(unavailable),
            "total_examples": total_examples,
            "weighted_loss": round(weighted_loss, 6),
        }
        raft_entry = self.propose_raft_tx("LORA_ROUND_FINALIZE", finalize_payload)

        return {
            "round_id": round_id,
            "lora_mode": lora_mode,
            "aggregation": aggregation,
            "server_lr": server_lr,
            "global_adapter_cid": global_cid,
            "global_adapter_sha256": global_sha,
            "participants": participants,
            "unavailable_members": unavailable,
            "total_examples": total_examples,
            "weighted_loss": round(weighted_loss, 6),
            "raft_block_index": raft_entry["index"],
            "raft_entry_hash": raft_entry["entry_hash"],
            "clients": clients,
        }

    # ------------------------------------------------------------- query path ---
    def query_decentralized_rag(self, body: dict) -> dict:
        """Scatter-gather over every member with a per-peer timeout, verifying ledger hashes."""
        query = body["query"]
        k = int(body.get("k", 3))
        hits = self.store.search_local(query, k=k)
        searched_nodes = [self.node_id]

        responses, failures = self._fan_out("/rag/search-local", {"query": query, "k": k})
        for peer_id, res in responses.items():
            hits.extend(res.get("results", []))
            searched_nodes.append(res.get("node_id") or peer_id)
        missing_nodes = sorted(failures)

        ledger_docs = self.get_raft_ledger().get("ledger", {}).get("documents", {})
        verified_hits = []
        for hit in hits:
            doc_record = ledger_docs.get(hit["doc_id"])
            hash_verified = bool(
                doc_record
                and doc_record.get("pdf_sha256") == hit["pdf_sha256"]
                and doc_record.get("chunks_sha256") == hit["chunks_sha256"]
            )
            verified_hits.append({**hit, "ledger_hash_verified": hash_verified})

        verified_hits.sort(key=lambda item: item["score"], reverse=True)
        return {
            "query": query,
            "top_k": verified_hits[:k],
            "searched_nodes": searched_nodes,
            "missing_nodes": missing_nodes,
            "missing_shards": self.missing_shards(missing_nodes),
            "complete": not missing_nodes,
            "peer_errors": failures,
        }


def build_handler(app: NodeApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            try:
                if self.path == "/embedding-model":
                    self._send(200, {"digest": app.store.embedder.digest})
                elif self.path == "/status":
                    self._send(
                        200,
                        {
                            "node_id": app.node_id,
                            "shard_id": app.shard_id,
                            "members": sorted(app.members),
                            "rag_store": app.store.summary(),
                            "ipfs": {
                                "mode": getattr(app.ipfs, "mode", "unknown"),
                                "pins": app.ipfs.pin_ls(),
                            },
                            "raft": app.get_raft_ledger(),
                        },
                    )
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(500, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/shard/ingest-full":
                    self._send(200, app.ingest_full_and_replicate(body))
                elif self.path == "/shard/ingest-stub":
                    self._send(200, app.store.ingest_metadata_stub(body))
                elif self.path == "/shard/store-backup":
                    self._send(200, app.store_backup_and_attest(body))
                elif self.path == "/shard/export-backup":
                    self._send(200, app.export_backup(body))
                elif self.path == "/shard/restore":
                    self._send(200, app.restore_owned_shard(body))
                elif self.path == "/rag/search-local":
                    hits = app.store.search_local(body["query"], k=int(body.get("k", 3)))
                    self._send(200, {"node_id": app.node_id, "results": hits})
                elif self.path == "/rag/query-decentralized":
                    self._send(200, app.query_decentralized_rag(body))
                elif self.path == "/fed/train-local":
                    self._send(200, app.train_local_round(body))
                elif self.path == "/fed/run-round":
                    self._send(200, app.coordinate_federated_round(body))
                else:
                    self._send(404, {"error": f"unknown endpoint {self.path}"})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

    return Handler


def select_embedder():
    mode = os.environ.get("FEDRAG_EMBEDDER", "hash").lower()
    if mode == "ollama":
        try:
            emb = OllamaEmbedder()
            emb.embed(["warmup test"])
            return emb
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc
    return HashEmbedder()


def select_ipfs_client(node_id: str):
    api_url = os.environ.get("FEDRAG_IPFS_API_URL", "").strip()
    if api_url:
        client = KuboIPFSClient(api_url=api_url)
        try:
            client.id_info()
            return client
        except Exception as exc:
            print(f"[node_service] Kubo at {api_url} not ready yet ({exc}), using KuboIPFSClient lazy mode", flush=True)
            return client
    return LocalCIDStore(node_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="FabricFedRAG mTLS Node Service")
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--members",
        default="",
        help="RAG members as node_id=url pairs (defines the shard ring); include this node",
    )
    parser.add_argument("--listen-port", type=int, default=8443)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--raft-url", required=True)
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    members = parse_member_urls(args.members)
    if args.node_id not in members:
        raise SystemExit(f"--members must include this node {args.node_id!r}: got {sorted(members)}")

    embedder = select_embedder()
    ipfs_client = select_ipfs_client(args.node_id)
    app = NodeApp(
        node_id=args.node_id,
        data_dir=Path(args.data_dir),
        raft_url=args.raft_url,
        members=members,
        ca_file=args.ca,
        cert_file=args.cert,
        key_file=args.key,
        embedder=embedder,
        ipfs_client=ipfs_client,
    )

    server = ThreadingHTTPServer(("0.0.0.0", args.listen_port), build_handler(app))
    server.socket = make_server_ssl_context(args.ca, args.cert, args.key).wrap_socket(
        server.socket, server_side=True
    )
    print(
        f"[node_service] {args.node_id} ({app.shard_id} of {len(members)}) listening with mTLS on "
        f":{args.listen_port} (embedder={getattr(embedder, 'model', type(embedder).__name__)}, "
        f"members={sorted(members)})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
