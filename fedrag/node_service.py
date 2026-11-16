"""Per-Node mTLS Service for Decentralized 50/50 Sharded RAG & Federated LoRA Training.

Runs on each Docker peer (`node1-maestro` and `node2-peer`), enforcing TLS 1.3
mutual authentication (`ssl.CERT_REQUIRED`) for all peer-to-peer and Maestro-to-peer calls,
and committing all document hashes, backup verifications, and LoRA round hashes to the
Go Raft ledger (`cmd/raftnode`).
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from fedrag.embed import HashEmbedder, OllamaEmbedder
from fedrag.fedavg import adapter_sha256, average_adapters, load_adapter_npz, save_adapter_npz
from fedrag.shard_store import ShardedRAGStore
from fedrag.trainer import train_local_lora_round


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
        shard_id: str,
        data_dir: Path,
        raft_url: str,
        peer_url: str,
        ca_file: str,
        cert_file: str,
        key_file: str,
        embedder,
    ) -> None:
        self.node_id = node_id
        self.shard_id = shard_id
        self.data_dir = Path(data_dir)
        self.raft_url = raft_url.rstrip("/")
        self.peer_url = peer_url.rstrip("/") if peer_url else ""
        self.ssl_client_ctx = make_client_ssl_context(ca_file, cert_file, key_file)
        self.store = ShardedRAGStore(node_id=node_id, root_dir=self.data_dir / "rag", embedder=embedder)
        self.rounds_dir = self.data_dir / "rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)

    def propose_raft_tx(self, tx_type: str, payload: dict) -> dict:
        return mtls_request(
            self.ssl_client_ctx,
            f"{self.raft_url}/tx/propose",
            method="POST",
            payload={"tx_type": tx_type, "payload": payload},
        )

    def get_raft_ledger(self) -> dict:
        return mtls_request(self.ssl_client_ctx, f"{self.raft_url}/ledger", method="GET")

    def ingest_full_and_replicate(self, body: dict) -> dict:
        pdf_bytes = base64.b64decode(body["pdf_base64"].encode("ascii"))
        metadata_holder = body.get("metadata_holder") or "node2-peer"
        manifest = self.store.ingest_full_document(
            doc_id=body["doc_id"],
            title=body["title"],
            shard_owner=self.node_id,
            metadata_holder=metadata_holder,
            pdf_bytes=pdf_bytes,
            chunks=body["chunks"],
            page_count=int(body.get("page_count", 1)),
            training_pairs=body.get("training_pairs", []),
        )
        # 1. Commit document manifest + SHA-256 hashes to Go Raft ledger
        raft_entry = self.propose_raft_tx("DOC_MANIFEST", manifest)

        # 2. Send metadata-only stub (with SHA-256 hash) to the other peer over mTLS
        stub_ack = None
        backup_ack = None
        if self.peer_url:
            stub_ack = mtls_request(
                self.ssl_client_ctx,
                f"{self.peer_url}/shard/ingest-stub",
                method="POST",
                payload=manifest,
            )
            # 3. Send backup PDF replica to the other peer for checksum verification & backup storage
            backup_ack = mtls_request(
                self.ssl_client_ctx,
                f"{self.peer_url}/shard/store-backup",
                method="POST",
                payload={
                    "doc_id": manifest["doc_id"],
                    "source_node": self.node_id,
                    "pdf_base64": body["pdf_base64"],
                    "expected_sha256": manifest["pdf_sha256"],
                },
            )
        return {
            "manifest": manifest,
            "raft_block_index": raft_entry["index"],
            "raft_entry_hash": raft_entry["entry_hash"],
            "peer_metadata_stub": stub_ack,
            "peer_backup_attestation": backup_ack,
        }

    def store_backup_and_attest(self, body: dict) -> dict:
        pdf_bytes = base64.b64decode(body["pdf_base64"].encode("ascii"))
        attestation = self.store.store_and_verify_backup(
            doc_id=body["doc_id"],
            source_node=body["source_node"],
            pdf_bytes=pdf_bytes,
            expected_sha256=body["expected_sha256"],
        )
        raft_entry = self.propose_raft_tx("BACKUP_CHECK", attestation)
        return {**attestation, "raft_block_index": raft_entry["index"]}

    def train_local_round(self, body: dict) -> dict:
        round_id = int(body["round_id"])
        round_dir = self.rounds_dir / f"round-{round_id}" / self.node_id
        global_dir = None
        if body.get("global_adapter_b64"):
            global_state = deserialize_state_b64(body["global_adapter_b64"])
            global_dir = self.rounds_dir / f"round-{round_id}" / "global-init"
            save_adapter_npz(global_dir, global_state, {"round_id": round_id - 1})

        examples = self.store.export_training_shard()
        result = train_local_lora_round(
            node_id=self.node_id,
            shard_id=self.shard_id,
            round_id=round_id,
            examples=examples,
            out_dir=round_dir,
            global_adapter_dir=global_dir,
        )
        state, _ = load_adapter_npz(round_dir)

        submission = {
            "round_id": round_id,
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "adapter_sha256": result["adapter_sha256"],
            "num_examples": result["num_examples"],
            "first_loss": result["first_loss"],
            "last_loss": result["last_loss"],
        }
        raft_entry = self.propose_raft_tx("LORA_CLIENT_SUBMIT", submission)
        return {
            **result,
            "raft_block_index": raft_entry["index"],
            "adapter_b64": serialize_state_b64(state),
        }

    def coordinate_federated_round(self, body: dict) -> dict:
        """Maestro coordinator: train on local Shard 1 + remote Shard 2 over mTLS, verify hashes, run FedAvg."""
        round_id = int(body["round_id"])
        global_b64 = None
        if round_id > 1:
            prev_global = self.rounds_dir / f"round-{round_id - 1}" / "global-final"
            if prev_global.exists():
                prev_state, _ = load_adapter_npz(prev_global)
                global_b64 = serialize_state_b64(prev_state)

        # 1. Train on Maestro's local Shard 1
        local_res = self.train_local_round({"round_id": round_id, "global_adapter_b64": global_b64})
        local_state = deserialize_state_b64(local_res["adapter_b64"])

        # 2. Train on Peer's Shard 2 over mTLS
        peer_res = mtls_request(
            self.ssl_client_ctx,
            f"{self.peer_url}/fed/train-local",
            method="POST",
            payload={"round_id": round_id, "global_adapter_b64": global_b64},
        )
        peer_state = deserialize_state_b64(peer_res["adapter_b64"])

        # 3. Verify that received LoRA tensors match their SHA-256 commitments
        if adapter_sha256(local_state) != local_res["adapter_sha256"]:
            raise RuntimeError("Local LoRA adapter SHA-256 verification failed")
        if adapter_sha256(peer_state) != peer_res["adapter_sha256"]:
            raise RuntimeError("Peer LoRA adapter SHA-256 verification failed")

        # 4. Weighted FedAvg across both shards
        global_state = average_adapters(
            [
                (local_state, float(local_res["num_examples"])),
                (peer_state, float(peer_res["num_examples"])),
            ]
        )
        global_dir = self.rounds_dir / f"round-{round_id}" / "global-final"
        total_examples = int(local_res["num_examples"]) + int(peer_res["num_examples"])
        weighted_loss = (
            float(local_res["last_loss"]) * int(local_res["num_examples"])
            + float(peer_res["last_loss"]) * int(peer_res["num_examples"])
        ) / total_examples

        global_sha = save_adapter_npz(
            global_dir,
            global_state,
            {
                "round_id": round_id,
                "total_examples": total_examples,
                "weighted_loss": round(weighted_loss, 6),
            },
        )

        # 5. Commit LORA_ROUND_FINALIZE to Go Raft ledger (chaincode verifies client hashes)
        finalize_payload = {
            "round_id": round_id,
            "global_adapter_sha256": global_sha,
            "participant_hashes": {
                local_res["node_id"]: local_res["adapter_sha256"],
                peer_res["node_id"]: peer_res["adapter_sha256"],
            },
            "total_examples": total_examples,
            "weighted_loss": round(weighted_loss, 6),
        }
        raft_entry = self.propose_raft_tx("LORA_ROUND_FINALIZE", finalize_payload)

        return {
            "round_id": round_id,
            "global_adapter_sha256": global_sha,
            "total_examples": total_examples,
            "weighted_loss": round(weighted_loss, 6),
            "raft_block_index": raft_entry["index"],
            "raft_entry_hash": raft_entry["entry_hash"],
            "clients": [
                {k: v for k, v in local_res.items() if k != "adapter_b64"},
                {k: v for k, v in peer_res.items() if k != "adapter_b64"},
            ],
        }

    def query_decentralized_rag(self, body: dict) -> dict:
        """Scatter-gather query across Shard 1 (local) and Shard 2 (peer over mTLS), verifying ledger hashes."""
        query = body["query"]
        k = int(body.get("k", 3))
        local_hits = self.store.search_local(query, k=k)
        peer_hits = []
        if self.peer_url:
            res = mtls_request(
                self.ssl_client_ctx,
                f"{self.peer_url}/rag/search-local",
                method="POST",
                payload={"query": query, "k": k},
            )
            peer_hits = res.get("results", [])

        ledger_docs = self.get_raft_ledger().get("ledger", {}).get("documents", {})
        verified_hits = []
        for hit in local_hits + peer_hits:
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
            "searched_nodes": [self.node_id] + ([ "node2-peer" ] if self.peer_url else []),
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
                if self.path == "/status":
                    self._send(
                        200,
                        {
                            "rag_store": app.store.summary(),
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
            print(f"[node_service] Ollama unavailable ({exc}), falling back to HashEmbedder", flush=True)
    return HashEmbedder()


def main() -> None:
    parser = argparse.ArgumentParser(description="FabricFedRAG mTLS Node Service")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--listen-port", type=int, default=8443)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--raft-url", required=True)
    parser.add_argument("--peer-url", default="")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    embedder = select_embedder()
    app = NodeApp(
        node_id=args.node_id,
        shard_id=args.shard_id,
        data_dir=Path(args.data_dir),
        raft_url=args.raft_url,
        peer_url=args.peer_url,
        ca_file=args.ca,
        cert_file=args.cert,
        key_file=args.key,
        embedder=embedder,
    )

    server = ThreadingHTTPServer(("0.0.0.0", args.listen_port), build_handler(app))
    server.socket = make_server_ssl_context(args.ca, args.cert, args.key).wrap_socket(
        server.socket, server_side=True
    )
    print(
        f"[node_service] {args.node_id} ({args.shard_id}) listening with mTLS on :{args.listen_port} "
        f"(embedder={getattr(embedder, 'model', type(embedder).__name__)})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
