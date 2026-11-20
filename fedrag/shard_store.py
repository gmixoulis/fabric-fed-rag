"""N-way Sharded PDF + Embedding RAG Store with SHA-256 Metadata Stubs & Backup Verification.

Shard assignment is a seeded, deterministic hash of `doc_id` over the ordered member list
(`ShardRing`): with three members each node owns a third, with two members the ring is the
original 50/50 split, unchanged.

Each node holds:
- Full PDF binary + text chunks + embeddings + training pairs for the shard it owns.
- Metadata-only stub + verified SHA-256 hashes (`pdf_sha256`, `chunks_sha256`) for every
  other shard.
- ONE backup replica of the next node's shard (ring: 1 backs up 2, 2 backs up 3, 3 backs
  up 1; with two nodes each backs up the other), verified against the ledger checksums.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fedrag.embed import top_k
from fedrag.ipfs_client import compute_cidv1_raw
from fedrag.pdf_embedder import (
    PDFEmbedder,
    build_metadata_stub_pdf,
    compute_chunks_sha256,
    compute_content_sha256,
    extract_and_validate_pdf,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    shard_owner     TEXT NOT NULL,
    metadata_holder TEXT NOT NULL,
    ipfs_cid        TEXT NOT NULL DEFAULT '',
    pdf_sha256      TEXT NOT NULL,
    content_sha256  TEXT NOT NULL DEFAULT '',
    chunks_sha256   TEXT NOT NULL,
    stub_pdf_sha256 TEXT NOT NULL DEFAULT '',
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


SHARD_SALT = "fabric-fed-rag/shard-ring/v1"


def shard_index_for(doc_id: str, num_shards: int, salt: str = SHARD_SALT) -> int:
    """Seeded, deterministic shard index for one document id.

    The salt is part of the hash input so a different split can be reproduced later
    without moving documents that were already committed to the ledger.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    digest = hashlib.sha256(f"{salt}:{doc_id}".encode()).digest()
    return int.from_bytes(digest, "big") % num_shards


class ShardRing:
    """Deterministic hash shard ring over the consortium members.

    The ring order is the sorted member id list, so every node derives the same shard
    index, owner and backup target for the same document without any extra state.
    With two members this is exactly the old 50/50 split: each node owns one half and
    holds the backup of the other half.
    """

    def __init__(self, members: Iterable[str], salt: str = SHARD_SALT) -> None:
        if isinstance(members, str):
            raise TypeError("ShardRing takes a sequence of member ids, not one string")
        ordered = sorted({str(member).strip() for member in members if str(member).strip()})
        if not ordered:
            raise ValueError("a shard ring needs at least one member")
        self.members: tuple[str, ...] = tuple(ordered)
        self.salt = salt

    @property
    def num_shards(self) -> int:
        return len(self.members)

    def shard_id(self, index: int) -> str:
        if not 0 <= index < self.num_shards:
            raise ValueError(f"shard index {index} is outside 0..{self.num_shards - 1}")
        return f"shard-{index + 1}"

    def index_of_node(self, node_id: str) -> int:
        try:
            return self.members.index(node_id)
        except ValueError:
            raise ValueError(f"{node_id!r} is not a ring member") from None

    def index(self, doc_id: str) -> int:
        return shard_index_for(doc_id, self.num_shards, self.salt)

    def owner(self, doc_id: str) -> str:
        return self.members[self.index(doc_id)]

    def backup_index(self, index: int) -> int:
        if self.num_shards < 2:
            raise ValueError("a backup replica needs at least two ring members")
        return (index + 1) % self.num_shards

    def backup_of_index(self, index: int) -> str:
        return self.members[self.backup_index(index)]

    def backup(self, doc_id: str) -> str:
        return self.backup_of_index(self.index(doc_id))

    def stub_holders(self, doc_id: str) -> list[str]:
        """Every member except the owner keeps a metadata-only stub for this document."""
        owner = self.owner(doc_id)
        return [member for member in self.members if member != owner]

    def assignment(self, doc_id: str) -> dict:
        index = self.index(doc_id)
        return {
            "doc_id": doc_id,
            "shard_index": index,
            "shard_id": self.shard_id(index),
            "owner": self.members[index],
            "backup": self.backup_of_index(index) if self.num_shards > 1 else None,
        }

    def balance(self, doc_ids: Iterable[str]) -> dict[str, int]:
        counts = {self.shard_id(index): 0 for index in range(self.num_shards)}
        for doc_id in doc_ids:
            counts[self.shard_id(self.index(doc_id))] += 1
        return counts


class ShardedRAGStore:
    """SQLite + filesystem store for one peer node in the sharded RAG network."""

    def __init__(
        self,
        node_id: str,
        root_dir: Path | str,
        embedder,
        *,
        members: Iterable[str] | None = None,
        shard_salt: str = SHARD_SALT,
    ) -> None:
        self.node_id = node_id
        self.root_dir = Path(root_dir)
        self.pdfs_dir = self.root_dir / "pdfs"
        self.backups_dir = self.root_dir / "backups"
        self.db_path = self.root_dir / "rag_shard.sqlite"
        self.embedder = embedder
        self.pdf_embedder = PDFEmbedder(embedder)
        # Without a configured consortium this store is a single-shard ring that owns
        # every document (legacy callers and offline fixtures keep working).
        self.ring = ShardRing(members if members is not None else [node_id], shard_salt)
        if node_id not in self.ring.members:
            raise ValueError(f"node {node_id!r} is not in the shard ring {self.ring.members}")

        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def shard_index(self) -> int:
        return self.ring.index_of_node(self.node_id)

    @property
    def shard_id(self) -> str:
        return self.ring.shard_id(self.shard_index)

    def owns(self, doc_id: str) -> bool:
        return self.ring.owner(doc_id) == self.node_id

    def backup_target(self) -> str | None:
        """The node whose shard this node replicates, or None in a single-member ring."""
        if self.ring.num_shards < 2:
            return None
        return self.ring.backup_of_index(self.shard_index)

    def shard_info(self) -> dict:
        target = self.backup_target()
        return {
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "shard_count": self.ring.num_shards,
            "members": list(self.ring.members),
            "backup_target": target,
            "backup_target_shard": self.ring.shard_id(self.ring.index_of_node(target)) if target else None,
        }

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
        shard_owner: str | None = None,
        metadata_holder: str | None = None,
        pdf_bytes: bytes,
        chunks: list[str] | None = None,
        page_count: int = 1,
        ipfs_cid: str | None = None,
        training_pairs: list[dict[str, str]] | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """Validate PDF metadata hashes, extract/embed chunks, and store the document in its owner shard."""
        assigned = self.ring.index(doc_id)
        if self.ring.owner(doc_id) != self.node_id:
            raise ValueError(
                f"document {doc_id} belongs to {self.ring.shard_id(assigned)} owned by "
                f"{self.ring.owner(doc_id)}; this node owns {self.shard_id}"
            )
        resolved_owner = shard_owner or self.node_id
        if resolved_owner != self.node_id:
            raise ValueError(
                f"a store can only write documents it owns: shard_owner={resolved_owner!r}, "
                f"node={self.node_id!r}"
            )
        # The backup replica holder also carries the metadata-only stub for this document.
        resolved_holder = metadata_holder or self.backup_target() or self.node_id
        artifact = self.pdf_embedder.embed_pdf_bytes(
            pdf_bytes, fallback_chunks=chunks
        )
        resolved_chunks = chunks if chunks is not None else artifact.chunks
        pdf_hash = artifact.pdf_sha256
        content_hash = compute_content_sha256(resolved_chunks)
        chunks_hash = compute_chunks_sha256(resolved_chunks)
        cid = ipfs_cid or compute_cidv1_raw(pdf_bytes)

        pdf_file = self.pdfs_dir / f"{doc_id}.pdf"
        pdf_file.write_bytes(pdf_bytes)

        vectors = (
            self.embedder.embed(resolved_chunks)
            if resolved_chunks
            else np.zeros((0, 0), dtype=np.float64)
        )
        embed_model = self.pdf_embedder.model_name

        meta = {
            "doc_id": doc_id,
            "title": title,
            "page_count": page_count,
            "num_chunks": len(resolved_chunks),
            "ipfs_cid": cid,
            "pdf_sha256": pdf_hash,
            "content_sha256": content_hash,
            "chunks_sha256": chunks_hash,
            "shard_index": assigned,
            "shard_id": self.ring.shard_id(assigned),
            "shard_count": self.ring.num_shards,
            "backup_node": resolved_holder,
            "internal_hash_verified": artifact.internal_hash_verified,
            **(extra_metadata or {}),
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, title, shard_owner, metadata_holder, ipfs_cid, pdf_sha256,
                 content_sha256, chunks_sha256, stub_pdf_sha256, page_count, num_chunks,
                 is_metadata_only, pdf_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, 0, ?, ?)
                """,
                (
                    doc_id,
                    title,
                    resolved_owner,
                    resolved_holder,
                    cid,
                    pdf_hash,
                    content_hash,
                    chunks_hash,
                    page_count,
                    len(resolved_chunks),
                    str(pdf_file),
                    json.dumps(meta),
                ),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            for index, text in enumerate(resolved_chunks):
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
            "shard_owner": resolved_owner,
            "metadata_holder": resolved_holder,
            "shard_index": assigned,
            "shard_id": self.ring.shard_id(assigned),
            "shard_count": self.ring.num_shards,
            "backup_node": resolved_holder,
            "ipfs_cid": cid,
            "pdf_sha256": pdf_hash,
            "content_sha256": content_hash,
            "chunks_sha256": chunks_hash,
            "num_chunks": len(resolved_chunks),
            "page_count": page_count,
        }

    def ingest_metadata_stub(self, manifest: dict) -> dict:
        """Build & validate a Metadata-Only Stub PDF carrying the verified hashes for another shard."""
        doc_id = manifest["doc_id"]
        pdf_hash = manifest["pdf_sha256"]
        chunks_hash = manifest["chunks_sha256"]
        content_hash = manifest.get("content_sha256") or chunks_hash
        cid = manifest.get("ipfs_cid", "")
        if len(pdf_hash) != 64 or len(chunks_hash) != 64:
            raise ValueError(f"Invalid SHA-256 hash in metadata stub for {doc_id}")

        stub_bytes, stub_meta = build_metadata_stub_pdf(manifest)
        validated = extract_and_validate_pdf(
            stub_bytes, expected_pdf_sha256=stub_meta["stub_pdf_sha256"]
        )
        stub_file = self.pdfs_dir / f"{doc_id}.stub.pdf"
        stub_file.write_bytes(stub_bytes)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, title, shard_owner, metadata_holder, ipfs_cid, pdf_sha256,
                 content_sha256, chunks_sha256, stub_pdf_sha256, page_count, num_chunks,
                 is_metadata_only, pdf_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    doc_id,
                    manifest["title"],
                    manifest["shard_owner"],
                    manifest["metadata_holder"],
                    cid,
                    pdf_hash,
                    content_hash,
                    chunks_hash,
                    validated["stub_pdf_sha256"],
                    int(manifest["page_count"]),
                    int(manifest["num_chunks"]),
                    str(stub_file),
                    json.dumps({**manifest, **stub_meta}),
                ),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return {
            "doc_id": doc_id,
            "is_metadata_only": True,
            "ipfs_cid": cid,
            "pdf_sha256": pdf_hash,
            "content_sha256": content_hash,
            "chunks_sha256": chunks_hash,
            "stub_pdf_sha256": validated["stub_pdf_sha256"],
            "stub_pdf_path": str(stub_file),
            "internal_hash_verified": validated["internal_hash_verified"],
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

    def read_backup(self, doc_id: str) -> dict:
        """Return one verified backup replica, ready to be restored by the owning node.

        The replica is re-hashed and compared with both the `BACKUP_CHECK` commitment and
        the document manifest, so a corrupted backup can never be restored silently.
        """
        with self._connect() as conn:
            backup = conn.execute(
                "SELECT doc_id, source_node, ipfs_cid, backup_path, computed_sha256, verified "
                "FROM backups WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            manifest = conn.execute(
                "SELECT doc_id, title, shard_owner, metadata_holder, ipfs_cid, pdf_sha256, "
                "page_count FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if backup is None or not backup["verified"]:
            raise ValueError(f"No verified backup replica for {doc_id} on {self.node_id}")
        pdf_bytes = Path(backup["backup_path"]).read_bytes()
        computed = sha256_bytes(pdf_bytes)
        if computed != backup["computed_sha256"]:
            raise ValueError(
                f"Backup replica for {doc_id} is corrupted: computed={computed} != "
                f"recorded={backup['computed_sha256']}"
            )
        if manifest is not None and manifest["pdf_sha256"] != computed:
            raise ValueError(
                f"Backup replica for {doc_id} no longer matches its manifest: "
                f"computed={computed} != manifest={manifest['pdf_sha256']}"
            )
        return {
            "doc_id": doc_id,
            "source_node": backup["source_node"],
            "backup_node": self.node_id,
            "ipfs_cid": backup["ipfs_cid"] or (manifest["ipfs_cid"] if manifest is not None else ""),
            "pdf_sha256": computed,
            "title": manifest["title"] if manifest is not None else "",
            "page_count": int(manifest["page_count"]) if manifest is not None else 1,
            "shard_owner": manifest["shard_owner"] if manifest is not None else self.ring.owner(doc_id),
            "pdf_bytes": pdf_bytes,
        }

    def ingest_restored_backup(self, replica: dict) -> dict:
        """Owner-side restore: re-index a document from a verified backup replica export."""
        artifact = self.pdf_embedder.embed_pdf_bytes(
            replica["pdf_bytes"], expected_pdf_sha256=replica["pdf_sha256"]
        )
        return self.ingest_full_document(
            doc_id=replica["doc_id"],
            title=replica.get("title") or artifact.title or replica["doc_id"],
            pdf_bytes=replica["pdf_bytes"],
            chunks=artifact.chunks,
            page_count=int(replica.get("page_count", 1)),
            ipfs_cid=replica.get("ipfs_cid") or None,
        )

    def all_document_metadata(self) -> list[dict]:
        """Every document this node knows, whether it holds the PDF or only a stub."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, title, shard_owner, metadata_holder, ipfs_cid, pdf_sha256, "
                "content_sha256, chunks_sha256, page_count, num_chunks, is_metadata_only "
                "FROM documents ORDER BY doc_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def search_local(self, query: str, k: int = 3) -> list[dict]:
        """Search the RAG shard this node owns by cosine similarity."""
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
        """Return the prompt/target training examples belonging to the shard this node owns."""
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
                "SELECT doc_id, title, ipfs_cid, pdf_sha256, content_sha256, chunks_sha256, num_chunks "
                "FROM documents WHERE is_metadata_only = 0 ORDER BY doc_id"
            ).fetchall()
            stub_docs = conn.execute(
                "SELECT doc_id, title, ipfs_cid, pdf_sha256, stub_pdf_sha256, pdf_path, shard_owner "
                "FROM documents WHERE is_metadata_only = 1 ORDER BY doc_id"
            ).fetchall()
            backups = conn.execute(
                "SELECT doc_id, source_node, ipfs_cid, computed_sha256, verified "
                "FROM backups ORDER BY doc_id"
            ).fetchall()
            examples_count = conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
            known_doc_ids = [
                row["doc_id"] for row in conn.execute("SELECT doc_id FROM documents ORDER BY doc_id")
            ]
        return {
            "node_id": self.node_id,
            "shard": self.shard_info(),
            "shard_coverage": self.ring.balance(known_doc_ids),
            "full_shard_documents": [dict(r) for r in full_docs],
            "metadata_only_stubs": [dict(r) for r in stub_docs],
            "verified_backups": [dict(r) for r in backups],
            "local_training_examples": int(examples_count),
        }
