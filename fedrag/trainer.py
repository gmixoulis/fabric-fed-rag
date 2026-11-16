"""Local LoRA Adapter Trainer on a Peer's 50% Document Shard (FedAvg-LoRA & FFA-LoRA).

Supports two federated low-rank adaptation modes:
1. ``fedavg``: Standard Federated LoRA (trains and averages both ``lora_A`` and ``lora_B``).
2. ``ffa``: Federated Freeze-A LoRA (``arXiv:2403.12313``), which keeps the deterministically
   seeded ``lora_A`` projection frozen across all clients and rounds, training and averaging
   only ``lora_B`` to eliminate cross-term aggregation noise (``B_bar @ A_bar != mean(B_i @ A_i)``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fedrag.embed import HashEmbedder
from fedrag.fedavg import load_adapter_npz, normalise_state, save_adapter_npz

DEFAULT_LORA_CONFIG = {
    "layers": 4,
    "d_model": 128,
    "r": 4,
    "alpha": 8.0,
    "target_modules": ("q_proj", "v_proj"),
    "local_steps": 8,
    "lr": 0.05,
    "seed": 42,
    "lora_mode": "fedavg",  # "fedavg" or "ffa" (Freeze-A LoRA, arXiv:2403.12313)
}


def _tensor_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def init_lora_state(config: dict | None = None) -> dict[str, np.ndarray]:
    """Initialize LoRA A (deterministically seeded) and B (zeros) matrices with standard PEFT keys."""
    cfg = {**DEFAULT_LORA_CONFIG, **(config or {})}
    layers = int(cfg["layers"])
    d_model = int(cfg["d_model"])
    rank = int(cfg["r"])
    seed = int(cfg["seed"])
    state: dict[str, np.ndarray] = {}

    for layer in range(layers):
        for module in cfg["target_modules"]:
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            rng = np.random.default_rng(_tensor_seed(seed, a_key))
            state[a_key] = rng.normal(0.0, 1.0 / np.sqrt(d_model), size=(rank, d_model)).astype(
                np.float32
            )
            state[b_key] = np.zeros((d_model, rank), dtype=np.float32)
    return state


def _project_text(embedder: HashEmbedder, text: str, d_model: int, seed: int) -> np.ndarray:
    raw = embedder.embed([text])[0].astype(np.float32)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(raw.shape[0]), size=(raw.shape[0], d_model)).astype(
        np.float32
    )
    vec = raw @ proj
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def train_local_lora_round(
    *,
    node_id: str,
    shard_id: str,
    round_id: int,
    examples: list[dict[str, str]],
    out_dir: Path | str,
    global_adapter_dir: Path | str | None = None,
    config: dict | None = None,
) -> dict:
    """Train one Federated LoRA round on this peer's local 50% shard examples."""
    if not examples:
        raise ValueError(f"Node {node_id} has no training examples in shard {shard_id}")

    cfg = {**DEFAULT_LORA_CONFIG, **(config or {})}
    d_model = int(cfg["d_model"])
    rank = int(cfg["r"])
    scale = float(cfg["alpha"]) / float(rank)
    lr = float(cfg["lr"])
    local_steps = int(cfg["local_steps"])
    layers = int(cfg["layers"])
    lora_mode = str(cfg.get("lora_mode", "fedavg")).lower()
    if lora_mode not in ("fedavg", "ffa"):
        raise ValueError(f"Unsupported lora_mode: {lora_mode!r}")

    if global_adapter_dir is not None and Path(global_adapter_dir).exists():
        loaded_state, _ = load_adapter_npz(Path(global_adapter_dir))
        state = {k: v.copy() for k, v in normalise_state(loaded_state).items()}
    else:
        state = init_lora_state(cfg)

    embedder = HashEmbedder()
    encoded = [
        (
            _project_text(embedder, ex["prompt"], d_model, 101),
            _project_text(embedder, ex["target"], d_model, 202),
        )
        for ex in examples
    ]

    step_losses: list[float] = []
    for _step in range(local_steps):
        total_loss = 0.0
        grads = {k: np.zeros_like(v) for k, v in state.items()}

        for x_vec, y_vec in encoded:
            h = x_vec.copy()
            layer_inputs: list[np.ndarray] = []
            for layer in range(layers):
                layer_inputs.append(h.copy())
                delta = np.zeros_like(h)
                for module in cfg["target_modules"]:
                    prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
                    a_mat = state[f"{prefix}.lora_A.weight"]
                    b_mat = state[f"{prefix}.lora_B.weight"]
                    delta += scale * (b_mat @ (a_mat @ h))
                h = h + delta

            diff = h - y_vec
            loss = 0.5 * float(np.sum(diff * diff))
            total_loss += loss

            for layer in range(layers):
                h_in = layer_inputs[layer]
                for module in cfg["target_modules"]:
                    prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
                    a_key = f"{prefix}.lora_A.weight"
                    b_key = f"{prefix}.lora_B.weight"
                    a_mat = state[a_key]
                    b_mat = state[b_key]

                    ah = a_mat @ h_in
                    grad_b = scale * np.outer(diff, ah)
                    grads[b_key] += grad_b / len(encoded)
                    if lora_mode == "fedavg":
                        grad_a = scale * np.outer(b_mat.T @ diff, h_in)
                        grads[a_key] += grad_a / len(encoded)

        for k in state:
            if lora_mode == "ffa" and k.endswith(".lora_A.weight"):
                continue
            state[k] -= lr * grads[k]
        step_losses.append(total_loss / len(encoded))

    meta = {
        "node_id": node_id,
        "shard_id": shard_id,
        "round_id": round_id,
        "lora_mode": lora_mode,
        "num_examples": len(examples),
        "first_loss": round(float(step_losses[0]), 6),
        "last_loss": round(float(step_losses[-1]), 6),
        "step_losses": [round(float(v), 6) for v in step_losses],
    }
    digest = save_adapter_npz(Path(out_dir), state, meta)
    return {**meta, "adapter_sha256": digest, "adapter_dir": str(out_dir)}
