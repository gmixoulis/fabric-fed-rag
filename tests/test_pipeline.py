"""Offline pytest suite for 50/50 Sharded RAG, Raft Ledger SHA-256 verification, and Federated LoRA (FedAvg & FFA-LoRA)."""

from __future__ import annotations

import numpy as np
import pytest

from fedrag.certgen import generate_swarm_key
from fedrag.embed import HashEmbedder
from fedrag.fedavg import adapter_sha256, average_adapters, load_adapter_npz
from fedrag.ipfs_client import LocalCIDStore, compute_cidv1_raw
from fedrag.pdf_embedder import (
    PDFEmbedder,
    build_full_pdf_with_metadata,
    build_metadata_stub_pdf,
    extract_and_validate_pdf,
)
from fedrag.raft_ledger import LedgerState
from fedrag.shard_store import ShardedRAGStore
from fedrag.trainer import train_local_lora_round


def test_50_50_sharding_and_backup_verification(tmp_path):
    embedder = HashEmbedder()
    node1 = ShardedRAGStore("node1-maestro", tmp_path / "node1", embedder)
    node2 = ShardedRAGStore("node2-peer", tmp_path / "node2", embedder)

    pdf_a = b"%PDF-1.4 Shard A Document Content %%EOF"
    manifest_a = node1.ingest_full_document(
        doc_id="doc-a",
        title="Shard A Full Doc",
        shard_owner="node1-maestro",
        metadata_holder="node2-peer",
        pdf_bytes=pdf_a,
        chunks=["Customer records have SODTYPE = 13 in TRDR."],
        page_count=2,
        training_pairs=[
            {"prompt": "Customers count", "target": "SELECT COUNT(*) FROM TRDR WHERE SODTYPE=13;"}
        ],
    )

    stub_res = node2.ingest_metadata_stub(manifest_a)
    assert stub_res["is_metadata_only"] is True
    assert stub_res["pdf_sha256"] == manifest_a["pdf_sha256"]

    backup_res = node2.store_and_verify_backup(
        doc_id="doc-a",
        source_node="node1-maestro",
        pdf_bytes=pdf_a,
        expected_sha256=manifest_a["pdf_sha256"],
    )
    assert backup_res["verified"] is True

    with pytest.raises(ValueError, match="Backup checksum verification failed"):
        node2.store_and_verify_backup(
            doc_id="doc-a",
            source_node="node1-maestro",
            pdf_bytes=b"corrupted pdf bytes",
            expected_sha256=manifest_a["pdf_sha256"],
        )


def test_federated_lora_fedavg_and_ffa_modes(tmp_path):
    shard1_examples = [
        {"prompt": "How many customers?", "target": "SELECT COUNT(*) FROM TRDR WHERE SODTYPE = 13;"},
        {"prompt": "How many suppliers?", "target": "SELECT COUNT(*) FROM TRDR WHERE SODTYPE = 12;"},
    ]
    shard2_examples = [
        {
            "prompt": "Count sales invoices",
            "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1351;",
        },
        {
            "prompt": "Count purchase invoices",
            "target": "SELECT COUNT(*) FROM FINDOC WHERE SOSOURCE = 1251;",
        },
    ]

    r1_node1 = train_local_lora_round(
        node_id="node1-maestro",
        shard_id="shard-1",
        round_id=1,
        examples=shard1_examples,
        out_dir=tmp_path / "r1" / "node1",
        config={"lora_mode": "ffa"},
    )
    r1_node2 = train_local_lora_round(
        node_id="node2-peer",
        shard_id="shard-2",
        round_id=1,
        examples=shard2_examples,
        out_dir=tmp_path / "r1" / "node2",
        config={"lora_mode": "ffa"},
    )

    assert r1_node1["last_loss"] < r1_node1["first_loss"]
    assert r1_node2["last_loss"] < r1_node2["first_loss"]

    s1, _ = load_adapter_npz(tmp_path / "r1" / "node1")
    s2, _ = load_adapter_npz(tmp_path / "r1" / "node2")

    # In FFA-LoRA mode, lora_A must remain identical across both shards
    for key in s1:
        if key.endswith(".lora_A.weight"):
            assert np.allclose(s1[key], s2[key])

    avg = average_adapters([(s1, 2.0), (s2, 2.0)])
    digest = adapter_sha256(avg)
    assert len(digest) == 64


def test_raft_ledger_chaincode_validation(tmp_path):
    ledger = LedgerState(tmp_path / "ledger.json")
    doc_hash = "a" * 64
    chunk_hash = "b" * 64

    e1 = ledger.validate_and_build_entry(
        1,
        "DOC_MANIFEST",
        {
            "doc_id": "doc-001",
            "title": "Test Shard 1 PDF",
            "shard_owner": "node1-maestro",
            "metadata_holder": "node2-peer",
            "pdf_sha256": doc_hash,
            "chunks_sha256": chunk_hash,
            "num_chunks": 2,
            "page_count": 1,
        },
    )
    ledger.apply_entry(e1)

    with pytest.raises(ValueError, match="Backup checksum mismatch"):
        ledger.validate_and_build_entry(
            1,
            "BACKUP_CHECK",
            {
                "doc_id": "doc-001",
                "source_node": "node1-maestro",
                "backup_node": "node2-peer",
                "computed_sha256": "c" * 64,
            },
        )


def test_ipfs_cidv1_and_private_swarm_key(tmp_path):
    store = LocalCIDStore("node1")
    payload = b"%PDF-1.4 Private Kubo Swarm Test %%EOF"
    cid = store.add_bytes(payload, "doc.pdf", pin=True)
    assert cid.startswith("bafkrei")
    assert cid == compute_cidv1_raw(payload)
    assert store.cat_bytes(cid) == payload
    assert cid in store.pin_ls()

    swarm_key = generate_swarm_key(tmp_path)
    lines = swarm_key.read_text().strip().splitlines()
    assert lines[0] == "/key/swarm/psk/1.0.0/"
    assert lines[1] == "/base16/"
    assert len(lines[2]) == 64


def test_pdf_embedder_and_metadata_stub_pdf(tmp_path):
    chunks = [
        "Customers are traders with TRDR.SODTYPE 13.",
        "Suppliers are traders with TRDR.SODTYPE 12.",
    ]
    pdf_bytes, meta = build_full_pdf_with_metadata(
        doc_id="doc-pdf-1",
        title="SoftOne Trader Codes",
        shard_owner="node1-maestro",
        metadata_holder="node2-peer",
        chunks=chunks,
        page_count=2,
    )

    pdf_embedder = PDFEmbedder(HashEmbedder())
    artifact = pdf_embedder.embed_pdf_bytes(pdf_bytes, expected_pdf_sha256=meta["pdf_sha256"])
    assert artifact.internal_hash_verified is True
    assert artifact.chunks == chunks
    assert artifact.content_sha256 == meta["content_sha256"]
    assert artifact.chunks_sha256 == meta["chunks_sha256"]
    assert artifact.vectors.shape[0] == 2

    # Build the metadata-only stub PDF for the other 50% peer node and validate its hash
    stub_bytes, stub_meta = build_metadata_stub_pdf(meta)
    parsed_stub = extract_and_validate_pdf(
        stub_bytes, expected_pdf_sha256=stub_meta["stub_pdf_sha256"]
    )
    assert parsed_stub["is_metadata_only"] is True
    assert parsed_stub["pdf_sha256"] == meta["pdf_sha256"]
    assert parsed_stub["content_sha256"] == meta["content_sha256"]
    assert parsed_stub["chunks_sha256"] == meta["chunks_sha256"]

    # Tampering with a chunk inside the PDF binary must fail internal metadata hash validation
    tampered_bytes = pdf_bytes.replace(b"SODTYPE 13", b"SODTYPE 99")
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        extract_and_validate_pdf(tampered_bytes)
