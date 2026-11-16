"""Unit tests for LoRA key normalisation, shape checks, weighted FedAvg, and SHA-256 tamper detection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fedrag.fedavg import (
    adapter_sha256,
    assert_same_keys,
    assert_same_shapes,
    average_adapters,
    denormalise_lora_key,
    load_adapter_npz,
    normalise_lora_key,
    normalise_state,
    save_adapter_npz,
)


def test_key_normalisation_roundtrip():
    multimodal = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    text_only = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert normalise_lora_key(multimodal) == text_only
    assert denormalise_lora_key(text_only) == multimodal

    state = {multimodal: np.ones((4, 8), dtype=np.float32)}
    norm = normalise_state(state)
    assert list(norm.keys()) == [text_only]


def test_key_and_shape_mismatch_raises():
    s1 = {"k1": np.zeros((2, 4), dtype=np.float32)}
    s2 = {"k2": np.zeros((2, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="key mismatch"):
        assert_same_keys([s1, s2])

    s3 = {"k1": np.zeros((4, 2), dtype=np.float32)}
    with pytest.raises(ValueError, match="shape mismatch"):
        assert_same_shapes([s1, s3])


def test_weighted_fedavg_and_tamper_detection(tmp_path):
    a1 = {"layer0.lora_B.weight": np.full((4, 2), 2.0, dtype=np.float32)}
    a2 = {"layer0.lora_B.weight": np.full((4, 2), 8.0, dtype=np.float32)}

    avg = average_adapters([(a1, 1.0), (a2, 3.0)])
    # Weighted mean: (1*2 + 3*8) / 4 = 6.5
    assert np.allclose(avg["layer0.lora_B.weight"], 6.5)

    out_dir = tmp_path / "adapter"
    digest = save_adapter_npz(out_dir, avg, {"round_id": 1})
    assert digest == adapter_sha256(avg)

    loaded, meta = load_adapter_npz(out_dir)
    assert meta["adapter_sha256"] == digest
    assert np.allclose(loaded["layer0.lora_B.weight"], 6.5)

    # Corrupt meta.json expected sha256 to trigger tamper detection
    meta["adapter_sha256"] = "0" * 64
    (out_dir / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Tampered LoRA adapter"):
        load_adapter_npz(out_dir)
