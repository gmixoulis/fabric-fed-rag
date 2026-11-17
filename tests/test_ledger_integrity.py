"""Regression tests for ledger integrity in fedrag.raft_ledger.LedgerState.

Covers three reproduced bugs:
1. A second DOC_MANIFEST for an existing doc_id may only re-commit an identical payload.
2. Every SHA-256 hash field must be exactly 64 hex characters, not merely 64 characters.
3. A ledger file loaded from disk must verify its hash chain and rebuild derived state.
"""

from __future__ import annotations

import json

import pytest

from fedrag.raft_ledger import LedgerState

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


def _commit(ledger: LedgerState, tx_type: str, payload: dict) -> dict:
    entry = ledger.validate_and_build_entry(1, tx_type, payload)
    ledger.apply_entry(entry)
    return entry


def _doc_manifest(doc_id: str = "doc-1", pdf_sha256: str = HEX_A) -> dict:
    return {
        "doc_id": doc_id,
        "title": "Spec",
        "shard_owner": "node1",
        "metadata_holder": "node2",
        "ipfs_cid": "QmTest",
        "pdf_sha256": pdf_sha256,
        "content_sha256": HEX_B,
        "chunks_sha256": HEX_B,
        "num_chunks": 1,
        "page_count": 1,
    }


def test_doc_manifest_overwrite_rejected_and_identical_recommit_accepted(tmp_path):
    ledger = LedgerState(tmp_path / "ledger.json")
    manifest = _doc_manifest()
    _commit(ledger, "DOC_MANIFEST", manifest)

    with pytest.raises(ValueError, match="already recorded"):
        _commit(ledger, "DOC_MANIFEST", _doc_manifest(pdf_sha256=HEX_C))

    before = ledger.snapshot()["height"]
    _commit(ledger, "DOC_MANIFEST", _doc_manifest())
    assert ledger.snapshot()["height"] == before + 1
    assert ledger.documents["doc-1"] == manifest


def test_non_hex_hashes_rejected(tmp_path):
    ledger = LedgerState(tmp_path / "ledger.json")

    with pytest.raises(ValueError, match="64-char sha256"):
        ledger.validate("DOC_MANIFEST", _doc_manifest(pdf_sha256="Z" * 64))

    bad_chunks = _doc_manifest()
    bad_chunks["chunks_sha256"] = "Z" * 64
    with pytest.raises(ValueError, match="64-char sha256"):
        ledger.validate("DOC_MANIFEST", bad_chunks)

    with pytest.raises(ValueError, match="Invalid LORA_CLIENT_SUBMIT"):
        ledger.validate(
            "LORA_CLIENT_SUBMIT",
            {"round_id": 1, "node_id": "node1", "adapter_sha256": "Z" * 64, "num_examples": 10},
        )

    with pytest.raises(ValueError, match="global_adapter_sha256"):
        ledger.validate(
            "LORA_ROUND_FINALIZE",
            {"round_id": 1, "global_adapter_sha256": "Z" * 64, "participant_hashes": {"node1": HEX_A}},
        )

    with pytest.raises(ValueError, match="Invalid participant hash"):
        ledger.validate(
            "LORA_ROUND_FINALIZE",
            {"round_id": 1, "global_adapter_sha256": HEX_A, "participant_hashes": {"node1": "Z" * 64}},
        )

    _commit(ledger, "DOC_MANIFEST", _doc_manifest())
    with pytest.raises(ValueError, match="Backup checksum mismatch"):
        ledger.validate("BACKUP_CHECK", {"doc_id": "doc-1", "computed_sha256": "Z" * 64})


def test_tampered_payload_raises_on_load(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = LedgerState(path)
    _commit(ledger, "DOC_MANIFEST", _doc_manifest())

    data = json.loads(path.read_text())
    data["entries"][1]["payload"]["pdf_sha256"] = HEX_D
    path.write_text(json.dumps(data, indent=2))

    with pytest.raises(ValueError, match="Entry hash verification failed at index 1"):
        LedgerState(path)


def test_tampered_derived_dict_ignored_on_load(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = LedgerState(path)
    _commit(ledger, "DOC_MANIFEST", _doc_manifest())

    data = json.loads(path.read_text())
    data["documents"]["doc-1"]["pdf_sha256"] = HEX_D
    data["documents"]["doc-1"]["title"] = "tampered"
    path.write_text(json.dumps(data, indent=2))

    reloaded = LedgerState(path)
    assert reloaded.documents["doc-1"]["pdf_sha256"] == HEX_A
    assert reloaded.documents["doc-1"]["title"] == "Spec"


def test_untampered_file_round_trips(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = LedgerState(path)
    doc = _doc_manifest()
    _commit(ledger, "DOC_MANIFEST", doc)
    _commit(
        ledger,
        "BACKUP_CHECK",
        {"doc_id": "doc-1", "source_node": "node1", "backup_node": "node2", "computed_sha256": HEX_A},
    )
    _commit(
        ledger,
        "LORA_CLIENT_SUBMIT",
        {"round_id": 1, "node_id": "node1", "adapter_sha256": HEX_C, "num_examples": 10},
    )
    _commit(
        ledger,
        "LORA_ROUND_FINALIZE",
        {"round_id": 1, "global_adapter_sha256": HEX_D, "participant_hashes": {"node1": HEX_C}},
    )

    snap = ledger.snapshot()

    reloaded = LedgerState(path)
    assert reloaded.entries == snap["entries"]
    assert reloaded.documents == snap["documents"]
    assert reloaded.backups == snap["backups"]
    assert reloaded.client_submissions == snap["client_submissions"]
    assert reloaded.final_rounds == snap["final_rounds"]
