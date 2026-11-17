"""Offline tests for the DoRA mode of the numpy LoRA trainer (arXiv:2402.09353)."""

from __future__ import annotations

import numpy as np
import pytest

from fedrag.embed import HashEmbedder
from fedrag.fedavg import average_adapters, load_adapter_npz, save_adapter_npz
from fedrag.trainer import (
    MAGNITUDE_SUFFIX,
    apply_adapter,
    encode_example,
    init_lora_state,
    train_local_lora_round,
)

EXAMPLES = [
    {"prompt": "Inventory goods SODTYPE in MTRL", "target": "MTRL stores items"},
    {"prompt": "Customer table TRDR", "target": "TRDR holds traders"},
]
SMALL = {"layers": 2, "d_model": 16, "r": 2, "local_steps": 3}
DORA = {**SMALL, "lora_mode": "dora"}

# Captured from the unmodified trainer (before DoRA was added) with EXAMPLES and SMALL.
BASELINE = {
    "fedavg": (
        "9cb5181ddacdf13b599bb960ba818f939bba2cf4fde371b45d9849fae3f1f377",
        [1.13091, 0.661438, 0.40882],
    ),
    "ffa": (
        "9a7dbf4f0e8779880a513bce454ce6bff1d36c9403d36e6fa1f398901a5cd85e",
        [1.13091, 0.661438, 0.417808],
    ),
}


def _train(tmp_path, name, config, examples=EXAMPLES, global_dir=None):
    return train_local_lora_round(
        node_id="n1",
        shard_id="s1",
        round_id=0,
        examples=examples,
        out_dir=tmp_path / name,
        global_adapter_dir=global_dir,
        config=config,
    )


@pytest.mark.parametrize("mode", ["fedavg", "ffa"])
def test_fedavg_and_ffa_unchanged(tmp_path, mode):
    out = _train(tmp_path, mode, {**SMALL, "lora_mode": mode})
    digest, losses = BASELINE[mode]
    assert out["adapter_sha256"] == digest
    assert out["step_losses"] == losses
    state, _ = load_adapter_npz(tmp_path / mode)
    assert not any(k.endswith(MAGNITUDE_SUFFIX) for k in state)
    assert not any(k.endswith(MAGNITUDE_SUFFIX) for k in init_lora_state(SMALL))


def test_unsupported_mode_raises(tmp_path):
    with pytest.raises(ValueError):
        _train(tmp_path, "bad", {**SMALL, "lora_mode": "vera"})


def test_dora_zero_contribution_at_init():
    state = init_lora_state(DORA)
    x, _ = encode_example(HashEmbedder(), EXAMPLES[0], 16)
    np.testing.assert_array_equal(apply_adapter(state, x, DORA), x)
    # Detection by key: a fedavg-mode config still runs the DoRA forward.
    np.testing.assert_array_equal(apply_adapter(state, x, SMALL), x)


def test_dora_trains_magnitude_and_lowers_loss(tmp_path):
    out = _train(tmp_path, "dora", DORA)
    assert out["lora_mode"] == "dora"
    assert out["last_loss"] < out["first_loss"]
    state, _ = load_adapter_npz(tmp_path / "dora")
    mags = [k for k in state if k.endswith(MAGNITUDE_SUFFIX)]
    assert len(mags) == 2 * 2  # layers x target_modules
    for k in mags:
        assert state[k].shape == (16,)
    assert max(float(np.abs(state[k] - 1.0).max()) for k in mags) > 1e-4
    x, _ = encode_example(HashEmbedder(), EXAMPLES[0], 16)
    assert not np.allclose(apply_adapter(state, x, DORA), x)


def test_dora_is_deterministic(tmp_path):
    assert (
        _train(tmp_path, "a", DORA)["adapter_sha256"]
        == _train(tmp_path, "b", DORA)["adapter_sha256"]
    )


def test_dora_clients_average_and_seed_next_round(tmp_path):
    _train(tmp_path, "c1", DORA, examples=EXAMPLES[:1])
    _train(tmp_path, "c2", DORA, examples=EXAMPLES[1:])
    s1, _ = load_adapter_npz(tmp_path / "c1")
    s2, _ = load_adapter_npz(tmp_path / "c2")
    avg = average_adapters([(s1, 1.0), (s2, 1.0)])
    for k in s1:
        np.testing.assert_allclose(avg[k], (s1[k] + s2[k]) / 2, rtol=1e-6, atol=1e-7)
    save_adapter_npz(tmp_path / "global", avg, {"round_id": 0})

    nxt = _train(tmp_path, "r1", DORA, global_dir=tmp_path / "global")
    state, _ = load_adapter_npz(tmp_path / "r1")
    assert set(state) == set(avg)
    # The next round starts from the averaged adapter, not from a fresh init.
    expected_first = np.mean(
        [
            0.5 * float(np.sum((apply_adapter(avg, x, DORA) - y) ** 2))
            for x, y in (encode_example(HashEmbedder(), ex, 16) for ex in EXAMPLES)
        ]
    )
    assert nxt["first_loss"] == pytest.approx(expected_first, rel=1e-4)


def test_dora_gradients_match_finite_differences(tmp_path):
    # One layer, one module: the trainer's "final diff for every layer" shortcut is exact here.
    cfg = {
        "layers": 1,
        "d_model": 8,
        "r": 2,
        "alpha": 4.0,
        "target_modules": ("q_proj",),
        "local_steps": 1,
        "lr": 1.0,
        "lora_mode": "dora",
    }
    scale = cfg["alpha"] / cfg["r"]
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    a_key, b_key, m_key = (
        f"{prefix}.lora_A.weight",
        f"{prefix}.lora_B.weight",
        prefix + MAGNITUDE_SUFFIX,
    )
    rng = np.random.default_rng(0)
    state0 = init_lora_state(cfg)
    state0[b_key] = rng.normal(0.0, 0.3, size=state0[b_key].shape).astype(np.float32)
    state0[m_key] = (1.0 + rng.normal(0.0, 0.1, size=8)).astype(np.float32)
    save_adapter_npz(tmp_path / "g", state0, {})

    example = EXAMPLES[:1]
    _train(tmp_path, "step", cfg, examples=example, global_dir=tmp_path / "g")
    state1, _ = load_adapter_npz(tmp_path / "step")
    analytic = {k: (state0[k].astype(np.float64) - state1[k]) / cfg["lr"] for k in state0}

    x, y = (v.astype(np.float64) for v in encode_example(HashEmbedder(), example[0], 8))
    base = {k: v.astype(np.float64) for k, v in state0.items()}
    norm0 = np.linalg.norm(np.eye(8) + scale * base[b_key] @ base[a_key], axis=1)

    def loss(s):
        v = np.eye(8) + scale * s[b_key] @ s[a_key]
        h = x + s[m_key] / norm0 * (v @ x) - x  # norm held constant (detached)
        return 0.5 * float(np.sum((h - y) ** 2))

    eps = 1e-6
    for key in (m_key, b_key, a_key):
        numeric = np.zeros_like(base[key])
        for idx in np.ndindex(base[key].shape):
            plus = {**base, key: base[key].copy()}
            minus = {**base, key: base[key].copy()}
            plus[key][idx] += eps
            minus[key][idx] -= eps
            numeric[idx] = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(analytic[key], numeric, rtol=1e-3, atol=1e-5, err_msg=key)
