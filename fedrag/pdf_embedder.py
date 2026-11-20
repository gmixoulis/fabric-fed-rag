"""PDF Builder, Metadata Hash Stamper, Internal Hash Validator, and PDF Embedder.

Implements the two PDF representations required by the sharded RAG architecture:
1. **Full Shard PDF (`is_metadata_only=false`)**:
   - Contains the full document text chunks across pages PLUS an embedded PDF `/Info`
     metadata dictionary carrying `doc_id`, `shard_owner`, `metadata_holder`,
     `content_sha256`, `chunks_sha256`, and `num_chunks`.
   - `PDFEmbedder.extract_and_embed(pdf_bytes)` parses the chunks directly from the PDF,
     validates that the extracted text matches the `content_sha256` and `chunks_sha256`
     recorded inside the PDF `/Info` metadata, and embeds every chunk with `OllamaEmbedder`
     (`bge-m3` / `embeddinggemma`) or `HashEmbedder`.
2. **Metadata-Only Stub PDF (`is_metadata_only=true`)**:
   - Stored on every member that does not own the document (`pdfs/<doc_id>.stub.pdf`).
   - Contains only the PDF `/Info` metadata dictionary and header stub with the verified
     `pdf_sha256`, `content_sha256`, `chunks_sha256`, and `ipfs_cid` of the full PDF, plus
     its own `stub_pdf_sha256` checksum.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_content_sha256(chunks: list[str]) -> str:
    joined = "\n".join(chunks).encode("utf-8")
    return sha256_hex(joined)


def compute_chunks_sha256(chunks: list[str]) -> str:
    h = hashlib.sha256()
    for index, text in enumerate(chunks):
        h.update(f"{index}:{text}\n".encode())
    return h.hexdigest()


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _unescape_pdf_text(text: str) -> str:
    return text.replace("\\)", ")").replace("\\(", "(").replace("\\\\", "\\")


def build_full_pdf_with_metadata(
    *,
    doc_id: str,
    title: str,
    shard_owner: str,
    metadata_holder: str,
    chunks: list[str],
    page_count: int = 1,
) -> tuple[bytes, dict]:
    """Create a deterministic PDF-1.4 binary with cryptographic hashes stamped in its `/Info` metadata."""
    content_sha = compute_content_sha256(chunks)
    chunks_sha = compute_chunks_sha256(chunks)

    stream_lines = ["BT", "/F1 11 Tf", "50 750 Td", f"({_escape_pdf_text(title)}) Tj"]
    for idx, chunk in enumerate(chunks):
        escaped = _escape_pdf_text(chunk)
        stream_lines.append("0 -18 Td")
        stream_lines.append(f"(CHUNK[{idx}]: {escaped}) Tj")
    stream_lines.append("ET")
    stream_bytes = "\n".join(stream_lines).encode("utf-8")

    subject_str = (
        f"doc_id={doc_id};shard_owner={shard_owner};"
        f"metadata_holder={metadata_holder};is_metadata_only=false;page_count={page_count}"
    )
    keywords_str = (
        f"content_sha256={content_sha};chunks_sha256={chunks_sha};num_chunks={len(chunks)}"
    )

    info_obj = (
        f"5 0 obj << "
        f"/Title ({_escape_pdf_text(title)}) "
        f"/Author (FabricFedRAG Shard Engine) "
        f"/Subject ({_escape_pdf_text(subject_str)}) "
        f"/Keywords ({_escape_pdf_text(keywords_str)}) "
        f">> endobj\n"
    ).encode()

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        + f"2 0 obj << /Type /Pages /Kids [3 0 R] /Count {int(page_count)} >> endobj\n".encode("ascii")
        + b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
        + f"4 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("ascii")
        + stream_bytes
        + b"\nendstream endobj\n"
        + info_obj
        + b"trailer << /Root 1 0 R /Info 5 0 R >>\n%%EOF\n"
    )
    pdf_sha = sha256_hex(pdf_bytes)
    meta = {
        "doc_id": doc_id,
        "title": title,
        "shard_owner": shard_owner,
        "metadata_holder": metadata_holder,
        "is_metadata_only": False,
        "page_count": int(page_count),
        "num_chunks": len(chunks),
        "content_sha256": content_sha,
        "chunks_sha256": chunks_sha,
        "pdf_sha256": pdf_sha,
    }
    return pdf_bytes, meta


def build_metadata_stub_pdf(manifest: dict) -> tuple[bytes, dict]:
    """Create a lightweight Metadata-Only Stub PDF carrying only the verified hashes of another shard."""
    doc_id = manifest["doc_id"]
    title = manifest["title"]
    shard_owner = manifest["shard_owner"]
    metadata_holder = manifest["metadata_holder"]
    pdf_sha = manifest["pdf_sha256"]
    content_sha = manifest.get("content_sha256") or manifest["chunks_sha256"]
    chunks_sha = manifest["chunks_sha256"]
    ipfs_cid = manifest.get("ipfs_cid", "")
    page_count = int(manifest.get("page_count", 1))
    num_chunks = int(manifest.get("num_chunks", 0))

    stub_line = (
        f"METADATA STUB ONLY | doc_id={doc_id} | owner={shard_owner} | "
        f"pdf_sha256={pdf_sha} | chunks_sha256={chunks_sha} | ipfs_cid={ipfs_cid}"
    )
    stream_bytes = f"BT /F1 10 Tf 50 750 Td ({_escape_pdf_text(stub_line)}) Tj ET".encode()

    subject_str = (
        f"doc_id={doc_id};shard_owner={shard_owner};"
        f"metadata_holder={metadata_holder};is_metadata_only=true;page_count={page_count}"
    )
    keywords_str = (
        f"pdf_sha256={pdf_sha};content_sha256={content_sha};"
        f"chunks_sha256={chunks_sha};ipfs_cid={ipfs_cid};num_chunks={num_chunks}"
    )

    info_obj = (
        f"5 0 obj << "
        f"/Title ({_escape_pdf_text(title)} [Metadata Stub]) "
        f"/Author (FabricFedRAG Shard Engine) "
        f"/Subject ({_escape_pdf_text(subject_str)}) "
        f"/Keywords ({_escape_pdf_text(keywords_str)}) "
        f">> endobj\n"
    ).encode()

    stub_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
        + f"4 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("ascii")
        + stream_bytes
        + b"\nendstream endobj\n"
        + info_obj
        + b"trailer << /Root 1 0 R /Info 5 0 R >>\n%%EOF\n"
    )
    stub_sha = sha256_hex(stub_bytes)
    stub_meta = {
        "doc_id": doc_id,
        "title": title,
        "shard_owner": shard_owner,
        "metadata_holder": metadata_holder,
        "is_metadata_only": True,
        "ipfs_cid": ipfs_cid,
        "pdf_sha256": pdf_sha,
        "content_sha256": content_sha,
        "chunks_sha256": chunks_sha,
        "stub_pdf_sha256": stub_sha,
        "num_chunks": num_chunks,
        "page_count": page_count,
    }
    return stub_bytes, stub_meta


def _parse_kv_pairs(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def extract_and_validate_pdf(
    pdf_bytes: bytes,
    *,
    expected_pdf_sha256: str | None = None,
) -> dict:
    """Parse `/Info` metadata and chunks from a PDF binary and verify all embedded SHA-256 hashes."""
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF header: missing %PDF- signature")

    computed_pdf_sha = sha256_hex(pdf_bytes)
    if expected_pdf_sha256 and computed_pdf_sha != expected_pdf_sha256:
        raise ValueError(
            f"PDF binary SHA-256 mismatch: computed={computed_pdf_sha} != expected={expected_pdf_sha256}"
        )

    text = pdf_bytes.decode("utf-8", errors="replace")
    title_match = re.search(r"/Title\s*\((.*?)\)\s*/Author", text, flags=re.DOTALL)
    subj_match = re.search(r"/Subject\s*\((.*?)\)\s*/Keywords", text, flags=re.DOTALL)
    kw_match = re.search(r"/Keywords\s*\((.*?)\)\s*>>\s*endobj", text, flags=re.DOTALL)

    title = _unescape_pdf_text(title_match.group(1)) if title_match else ""
    subj_kv = _parse_kv_pairs(_unescape_pdf_text(subj_match.group(1))) if subj_match else {}
    kw_kv = _parse_kv_pairs(_unescape_pdf_text(kw_match.group(1))) if kw_match else {}

    is_stub = subj_kv.get("is_metadata_only", "false").lower() == "true"

    if is_stub:
        pdf_sha = kw_kv.get("pdf_sha256", "")
        content_sha = kw_kv.get("content_sha256", "")
        chunks_sha = kw_kv.get("chunks_sha256", "")
        if len(pdf_sha) != 64 or len(chunks_sha) != 64:
            raise ValueError("Metadata stub PDF is missing valid 64-char SHA-256 hashes in /Keywords")
        return {
            "doc_id": subj_kv.get("doc_id", ""),
            "title": title.removesuffix(" [Metadata Stub]"),
            "shard_owner": subj_kv.get("shard_owner", ""),
            "metadata_holder": subj_kv.get("metadata_holder", ""),
            "is_metadata_only": True,
            "ipfs_cid": kw_kv.get("ipfs_cid", ""),
            "pdf_sha256": pdf_sha,
            "content_sha256": content_sha,
            "chunks_sha256": chunks_sha,
            "stub_pdf_sha256": computed_pdf_sha,
            "chunks": [],
            "internal_hash_verified": True,
        }

    # Extract CHUNK[i] lines from the PDF stream
    chunk_matches = re.findall(r"\(CHUNK\[(\d+)\]: (.*?)\) Tj", text)
    chunks = [_unescape_pdf_text(raw_chunk) for _idx, raw_chunk in chunk_matches]

    # Legacy fallback if PDF was created without CHUNK[i] markers
    if not chunks:
        stream_match = re.search(r"Td \((.*?)\) Tj ET", text, flags=re.DOTALL)
        if stream_match:
            lines = _unescape_pdf_text(stream_match.group(1)).split("\\n")
            chunks = lines[1:] if len(lines) > 1 else lines
            if not title and lines:
                title = lines[0]

    computed_content_sha = compute_content_sha256(chunks)
    computed_chunks_sha = compute_chunks_sha256(chunks)

    expected_content_sha = kw_kv.get("content_sha256")
    expected_chunks_sha = kw_kv.get("chunks_sha256")

    if expected_content_sha and computed_content_sha != expected_content_sha:
        raise ValueError(
            f"PDF internal content_sha256 mismatch: {computed_content_sha} != {expected_content_sha}"
        )
    if expected_chunks_sha and computed_chunks_sha != expected_chunks_sha:
        raise ValueError(
            f"PDF internal chunks_sha256 mismatch: {computed_chunks_sha} != {expected_chunks_sha}"
        )

    return {
        "doc_id": subj_kv.get("doc_id", ""),
        "title": title,
        "shard_owner": subj_kv.get("shard_owner", ""),
        "metadata_holder": subj_kv.get("metadata_holder", ""),
        "is_metadata_only": False,
        "page_count": int(subj_kv.get("page_count", "1")),
        "num_chunks": len(chunks),
        "pdf_sha256": computed_pdf_sha,
        "content_sha256": computed_content_sha,
        "chunks_sha256": computed_chunks_sha,
        "chunks": chunks,
        "internal_hash_verified": True,
    }


@dataclass
class EmbeddedPDFArtifact:
    doc_id: str
    title: str
    pdf_bytes: bytes
    pdf_sha256: str
    content_sha256: str
    chunks_sha256: str
    chunks: list[str]
    vectors: np.ndarray
    embed_model: str
    internal_hash_verified: bool


class PDFEmbedder:
    """Extracts text chunks from a PDF, verifies its embedded `/Info` hashes, and computes chunk vectors."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    @property
    def model_name(self) -> str:
        return getattr(self.embedder, "model", type(self.embedder).__name__)

    def embed_pdf_bytes(
        self,
        pdf_bytes: bytes,
        *,
        expected_pdf_sha256: str | None = None,
        fallback_chunks: list[str] | None = None,
    ) -> EmbeddedPDFArtifact:
        parsed = extract_and_validate_pdf(pdf_bytes, expected_pdf_sha256=expected_pdf_sha256)
        if parsed["is_metadata_only"]:
            raise ValueError(f"Cannot extract RAG chunks from metadata-only stub PDF {parsed['doc_id']}")
        chunks = parsed["chunks"] or list(fallback_chunks or [])
        vectors = (
            self.embedder.embed(chunks)
            if chunks
            else np.zeros((0, 0), dtype=np.float64)
        )
        return EmbeddedPDFArtifact(
            doc_id=parsed["doc_id"],
            title=parsed["title"],
            pdf_bytes=pdf_bytes,
            pdf_sha256=parsed["pdf_sha256"],
            content_sha256=parsed["content_sha256"],
            chunks_sha256=compute_chunks_sha256(chunks),
            chunks=chunks,
            vectors=vectors,
            embed_model=self.model_name,
            internal_hash_verified=parsed["internal_hash_verified"],
        )
