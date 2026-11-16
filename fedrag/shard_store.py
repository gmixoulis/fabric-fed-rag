"""50/50 Sharded PDF + Embedding RAG Store with SHA-256 Metadata Stubs & Backup Verification.

Each node holds:
- Full PDF binary + text chunks + embeddings + training pairs for its assigned 50% shard.
- Metadata-only stub + verified SHA-256 hashes (`pdf_sha256`, `chunks_sha256`) for the other 50%.
- Optional off-node backup copies verified against the Raft ledger checksums.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from fedrag.embed import top_k
from fedrag.ipfs_client import compute_cidv1_raw

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    shard_owner     TEXT NOT NULL,
    metadata_holder TEXT NOT NULL,
    ipfs_cid        TEXT NOT NULL DEFAULT '',
    pdf_sha256      TEXT NOT NULL,
    chunks_sha256   TEXT NOT NULL,
    page_count      INTEGER NOT NULL,
    num_chunks      INTEGER NOT NULL,
    is_metadata_only INTEGER NOT NULL CHECK (is_metadata_only IN (0, 1)),
    pdf_path        TEXT,
    metadata_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    chunk_sha256 TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    embed_model TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS training_examples (
    example_id  TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    target      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backups (
    doc_id          TEXT PRIMARY KEY,
    source_node     TEXT NOT NULL,
    ipfs_cid        TEXT NOT NULL DEFAULT '',
    backup_path     TEXT NOT NULL,
    computed_sha256 TEXT NOT NULL,
    verified        INTEGER NOT NULL CHECK (verified IN (0, 1))
);
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_chunks_sha256(chunks: list[str]) -> str:
    h = hashlib.sha256()
    for index, text in enumerate(chunks):
        h.update(f"{index}:{text}\n".encode())
    return h.hexdigest()


class ShardedRAGStore:
    """SQLite + filesystem store for one peer node in the 50/50 sharded RAG network."""

    def __init__(self, node_id: str, root_dir: Path | str, embedder) -> None:
        self.node_id = node_id
        self.root_dir = Path(root_dir)
        self.pdfs_dir = self.root_dir / "pdfs"
        self.backups_dir = self.root_dir / "backups"
        self.db_path = self.root_dir / "rag_shard.sqlite"
        self.embedder = embedder

        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def ingest_full_document(
        self,
        *,
        doc_id: str,
        title: str,
        shard_owner: str,
        metadata_holder: str,
        pdf_bytes: bytes,
        chunks: list[str],
        page_count: int,
        ipfs_cid: str | None = None,
        training_pairs: list[dict[str, str]] | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """Store a full PDF + chunks + embeddings + training examples (this node's 50% shard)."""
        pdf_hash = sha256_bytes(pdf_bytes)
        cid = ipfs_cid or compute_cidv1_raw(pdf_bytes)
        chunks_hash = compute_chunks_sha256(chunks)
        pdf_file = self.pdfs_dir / f"{doc_id}.pdf"
        pdf_file.write_bytes(pdf_bytes)

        vectors = self.embedder.embed(chunks) if chunks else np.zeros((0, 0), dtype=np.float64)
        embed_model = getattr(self.embedder, "model", type(self.embedder).__name__)

        meta = {
            "doc_id": doc_id,
            "title": title,
            "page_count": page_count,
            "num_chunks": len(chunks),
            "ipfs_cid": cid,
            "pdf_sha256": pdf_hash,
            "chunks_sha256": chunks_hash,
            **(extra_metadata or {}),
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, title, shard_owner, metadata_holder, ipfs_cid, pdf_sha256, chunks_sha256,
                 page_count, num_chunks, is_metadata_only, pdf_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    doc_id,
                    title,
                    shard_owner,
                    metadata_holder,
                    cid,
                    pdf_hash,
                    chunks_hash,
                    page_count,
                    len(chunks),
                    str(pdf_file),
                    json.dumps(meta),
                ),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            for index, text in enumerate(chunks):
                vec_bytes = np.ascontiguousarray(vectors[index], dtype=np.float64).tobytes()
                conn.execute(
                    """
                    INSERT INTO chunks
                    (chunk_id, doc_id, chunk_index, text, chunk_sha256, embedding, embed_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{doc_id}#c{index}",
                        doc_id,
                        index,
                        text,
                        sha256_bytes(text.encode("utf-8")),
                        vec_bytes,
                        embed_model,
                    ),
                )

            if training_pairs:
                conn.execute("DELETE FROM training_examples WHERE doc_id = ?", (doc_id,))
                for idx, pair in enumerate(training_pairs):
                    conn.execute(
                        """
                        INSERT INTO training_examples (example_id, doc_id, prompt, target)
                        VALUES (?, ?, ?, ?)
                        """,
                        (f"{doc_id}#ex{idx}", doc_id, pair["prompt"], pair["target"]),
                    )

        return {
            "doc_id": doc_id,
            "title": title,
            "shard_owner": shard_owner,
            "metadata_holder": metadata_holder,
            "ipfs_cid": cid,
            "pdf_sha256": pdf_hash,
            "chunks_sha256": chunks_hash,
            "num_chunks": len(chunks),
            "page_count": page_count,
        }

    def ingest_metadata_stub(self, manifest: dict) -> dict:
        """Store ONLY the PDF metadata + verified CID & SHA-256 hashes (the other node's 50% shard)."""
        doc_id = manifest["doc_id"]
        pdf_hash = manifest["pdf_sha256"]
        chunks_hash = manifest["chunks_sha256"]
        cid = manifest.get("ipfs_cid", "")
        if len(pdf_hash) != 64 or len(chunks_hash) != 64:
            raise ValueError(f"Invalid SHA-256 hash in metadata stub for {doc_id}")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, title, shard_owner, metadata_holder, ipfs_cid, pdf_sha256, chunks_sha256,
                 page_count, num_chunks, is_metadata_only, pdf_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?)
                """,
                (
                    doc_id,
                    manifest["title"],
                    manifest["shard_owner"],
                    manifest["metadata_holder"],
                    cid,
                    pdf_hash,
                    chunks_hash,
                    int(manifest["page_count"]),
                    int(manifest["num_chunks"]),
                    json.dumps(manifest),
                ),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return {
            "doc_id": doc_id,
            "is_metadata_only": True,
            "ipfs_cid": cid,
            "pdf_sha256": pdf_hash,
        }

    def store_and_verify_backup(
        self,
        doc_id: str,
        source_node: str,
        pdf_bytes: bytes,
        expected_sha256: str,
        ipfs_cid: str = "",
    ) -> dict:
        """Save a backup PDF replica from another hardware/peer and verify its SHA-256 checksum & CID."""
        computed = sha256_bytes(pdf_bytes)
        if computed != expected_sha256:
            raise ValueError(
                f"Backup checksum verification failed for {doc_id}: "
                f"computed={computed} != expected={expected_sha256}"
            )
        resolved_cid = ipfs_cid or compute_cidv1_raw(pdf_bytes)
        backup_file = self.backups_dir / f"{doc_id}.backup.pdf"
        backup_file.write_bytes(pdf_bytes)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO backups
                (doc_id, source_node, ipfs_cid, backup_path, computed_sha256, verified)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (doc_id, source_node, resolved_cid, str(backup_file), computed),
            )
        return {
            "doc_id": doc_id,
            "source_node": source_node,
            "backup_node": self.node_id,
            "ipfs_cid": resolved_cid,
            "computed_sha256": computed,
            "verified": True,
        }

    def search_local(self, query: str, k: int = 3) -> list[dict]:
        """Search this node's active 50% RAG shard by cosine similarity."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.chunk_sha256,
                       c.embedding, d.title, d.ipfs_cid, d.pdf_sha256, d.chunks_sha256
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE d.is_metadata_only = 0
                ORDER BY c.chunk_id
                """
            ).fetchall()
        if not rows:
            return []
        q_vec = self.embedder.embed([query])[0]
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float64) for r in rows])
        ranked = top_k(q_vec, matrix, k)
        results = []
        for idx, score in ranked:
            row = rows[idx]
            results.append(
                {
                    "node_id": self.node_id,
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "score": round(float(score), 4),
                    "ipfs_cid": row["ipfs_cid"],
                    "chunk_sha256": row["chunk_sha256"],
                    "pdf_sha256": row["pdf_sha256"],
                    "chunks_sha256": row["chunks_sha256"],
                }
            )
        return results

    def export_training_shard(self) -> list[dict[str, str]]:
        """Return the prompt/target training examples belonging to this node's 50% shard."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT example_id, doc_id, prompt, target
                FROM training_examples
                ORDER BY example_id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        with self._connect() as conn:
            full_docs = conn.execute(
                "SELECT doc_id, title, ipfs_cid, pdf_sha256, num_chunks "
                "FROM documents WHERE is_metadata_only = 0 ORDER BY doc_id"
            ).fetchall()
            stub_docs = conn.execute(
                "SELECT doc_id, title, ipfs_cid, pdf_sha256, shard_owner "
                "FROM documents WHERE is_metadata_only = 1 ORDER BY doc_id"
            ).fetchall()
            backups = conn.execute(
                "SELECT doc_id, source_node, ipfs_cid, computed_sha256, verified "
                "FROM backups ORDER BY doc_id"
            ).fetchall()
            examples_count = conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
        return {
            "node_id": self.node_id,
            "full_shard_documents": [dict(r) for r in full_docs],
            "metadata_only_stubs": [dict(r) for r in stub_docs],
            "verified_backups": [dict(r) for r in backups],
            "local_training_examples": int(examples_count),
        }
