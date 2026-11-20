"""LoRA key layout, exact shape validation, weighted FedAvg, and SHA-256 verification.

Ported from softone-rag/training/fed/fedavg.py and extended with deterministic
cryptographic hashing so the Python Raft ledger can verify every local and global adapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def normalise_lora_key(key: str) -> str:
    """Collapse the multimodal ``language_model`` segment into the standard text layout."""
    key = key.replace(".language_model.", ".")
    prefix = "language_model."
    if key.startswith(prefix):
        key = key[len(prefix):]
    return key


def denormalise_lora_key(key: str) -> str:
    """Inverse of :func:`normalise_lora_key` for multimodal text towers."""
    for prefix in ("base_model.model.model.", "model."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            if rest.startswith(("layers.", "embed_tokens.")):
                return prefix + "language_model." + rest
    return key


def normalise_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if any(".language_model." in key or key.startswith("language_model.") for key in state):
        return {normalise_lora_key(key): value for key, value in state.items()}
    return dict(state)


def assert_same_keys(states: list[dict[str, np.ndarray]]) -> list[str]:
    if not states:
        raise ValueError("No adapter states to compare")
    keys = sorted(states[0])
    for index, state in enumerate(states[1:], 2):
        other = sorted(state)
        if other != keys:
            missing = sorted(set(keys) - set(other))
            extra = sorted(set(other) - set(keys))
            raise ValueError(
                f"Adapter {index} key mismatch: missing={missing[:4]} extra={extra[:4]}"
            )
    return keys


def assert_same_shapes(states: list[dict[str, np.ndarray]]) -> list[str]:
    keys = assert_same_keys(states)
    reference = states[0]
    for index, state in enumerate(states[1:], 2):
        for key in keys:
            expected, actual = tuple(reference[key].shape), tuple(state[key].shape)
            if expected != actual:
                raise ValueError(
                    f"Adapter {index} shape mismatch for {key}: {actual} != {expected}"
                )
    return keys


def average_adapters(parts: list[tuple[dict[str, np.ndarray], float]]) -> dict[str, np.ndarray]:
    """Weighted element-wise mean of LoRA tensors; weights are example counts."""
    if not parts:
        raise ValueError("FedAvg needs at least one adapter")
    weights = [float(weight) for _, weight in parts]
    if any(weight < 0 for weight in weights):
        raise ValueError(f"FedAvg weights must be non-negative, got {weights}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("FedAvg weights sum to zero")
    states = [normalise_state(state) for state, _ in parts]
    keys = assert_same_shapes(states)
    averaged: dict[str, np.ndarray] = {}
    for key in keys:
        accumulated = None
        for state, weight in zip(states, weights, strict=True):
            term = np.asarray(state[key], dtype=np.float32) * np.float32(weight / total)
            accumulated = term if accumulated is None else accumulated + term
        averaged[key] = accumulated
    return averaged


AGGREGATION_MODES = ("average", "summed")
SERVER_LR_MAX = 16.0


def validate_server_lr(value: object) -> float:
    """The merge scale: 0 < server_lr <= 16, with 1.0 meaning plain FedAvg."""
    try:
        server_lr = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"server_lr must be a number, got {value!r}") from None
    if not 0.0 < server_lr <= SERVER_LR_MAX:
        raise ValueError(f"server_lr must be in (0, {SERVER_LR_MAX}], got {server_lr}")
    return server_lr


def server_lr_for(aggregation: str, num_participants: int) -> float:
    """Server learning rate for one aggregation mode.

    ``average`` is plain FedAvg (server_lr = 1). ``summed`` sums the per-node updates
    instead of averaging them: with disjoint shards that is a server learning rate equal
    to the number of participants (softone-rag E-2026-11-18).
    """
    mode = str(aggregation).lower()
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation {aggregation!r}; use one of {list(AGGREGATION_MODES)}")
    if num_participants < 1:
        raise ValueError("aggregation needs at least one participant")
    return 1.0 if mode == "average" else float(num_participants)


def aggregate_adapters(
    parts: list[tuple[dict[str, np.ndarray], float]],
    *,
    previous: dict[str, np.ndarray] | None = None,
    server_lr: float = 1.0,
) -> dict[str, np.ndarray]:
    """FedAvg, then one step of ``previous + server_lr * (weighted mean - previous)``.

    ``server_lr == 1`` returns the plain weighted mean untouched, so plain averaging is
    bit-identical to :func:`average_adapters`. ``server_lr == N`` over N disjoint shards
    sums their updates instead of halving them.
    """
    server_lr = validate_server_lr(server_lr)
    averaged = average_adapters(parts)
    if server_lr == 1.0:
        return averaged
    if previous is None:
        raise ValueError(
            f"server_lr {server_lr} needs the adapter the participants started this round from"
        )
    previous = normalise_state(previous)
    keys = assert_same_shapes([averaged, previous])
    return {key: previous[key] + server_lr * (averaged[key] - previous[key]) for key in keys}


def adapter_sha256(state: dict[str, np.ndarray]) -> str:
    """Compute a deterministic SHA-256 hash over sorted LoRA tensor names, shapes, and float32 bytes."""
    norm = normalise_state(state)
    hasher = hashlib.sha256()
    for key in sorted(norm):
        arr = np.ascontiguousarray(norm[key], dtype=np.float32)
        header = f"{key}:{list(arr.shape)}:float32\n".encode()
        hasher.update(header)
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def save_adapter_npz(path: Path, state: dict[str, np.ndarray], meta: dict) -> str:
    """Save normalised LoRA tensors and metadata, returning the deterministic SHA-256."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    norm = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in normalise_state(state).items()}
    digest = adapter_sha256(norm)
    np.savez(path / "adapter_model.npz", **norm)
    meta_out = {**meta, "adapter_sha256": digest, "num_tensors": len(norm)}
    (path / "meta.json").write_text(json.dumps(meta_out, indent=2))
    return digest


def load_adapter_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load LoRA tensors and verify their SHA-256 matches meta.json."""
    path = Path(path)
    with np.load(path / "adapter_model.npz") as data:
        state = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    meta = json.loads((path / "meta.json").read_text())
    computed = adapter_sha256(state)
    expected = meta.get("adapter_sha256")
    if expected and computed != expected:
        raise ValueError(f"Tampered LoRA adapter in {path}: sha256 {computed} != {expected}")
    return state, meta
