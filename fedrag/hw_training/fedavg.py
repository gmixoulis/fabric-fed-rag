"""LoRA key layout, exact key checks and weighted FedAvg for the federated run.

The PC base model is text-only (``model.layers.N...``) while the Mac checkpoint is
the multimodal Qwen3.5 repo, whose text tower lives at
``model.language_model.layers.N...``.  Adapters are always stored in the
text-model layout; the translation happens here and nowhere else, and every
average asserts that key sets and shapes match exactly before it adds numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

WEIGHT_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
META_FILE = "meta.json"


# --------------------------------------------------------------------------- #
# Key layout
# --------------------------------------------------------------------------- #
def normalise_lora_key(key: str) -> str:
    """Collapse the multimodal ``language_model`` segment into the text layout."""
    key = key.replace(".language_model.", ".")
    prefix = "language_model."
    if key.startswith(prefix):
        key = key[len(prefix):]
    return key


def denormalise_lora_key(key: str) -> str:
    """Inverse of :func:`normalise_lora_key` for the multimodal text tower."""
    for prefix in ("base_model.model.model.", "model."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            if rest.startswith(("layers.", "embed_tokens.")):
                return prefix + "language_model." + rest
    return key


def normalise_state(state: dict) -> dict:
    if any(".language_model." in key or key.startswith("language_model.") for key in state):
        return {normalise_lora_key(key): value for key, value in state.items()}
    return dict(state)


def assert_same_keys(states: list[dict]) -> list[str]:
    """Fail loudly when two adapters do not describe the same tensors."""
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


def assert_same_shapes(states: list[dict]) -> list[str]:
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


# --------------------------------------------------------------------------- #
# FedAvg
# --------------------------------------------------------------------------- #
def average_adapters(parts: list[tuple[dict, float]]) -> dict:
    """Weighted element-wise mean of LoRA tensors; weights are example counts."""
    if not parts:
        raise ValueError("FedAvg needs at least one adapter")
    weights = [float(weight) for _, weight in parts]
    if any(weight < 0 for weight in weights):
        raise ValueError(f"FedAvg weights must be non-negative, got {weights}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("FedAvg weights sum to zero")
    states = [state for state, _ in parts]
    keys = assert_same_shapes(states)
    averaged = {}
    for key in keys:
        accumulated = None
        for state, weight in zip(states, weights, strict=True):
            term = state[key] * (weight / total)
            accumulated = term if accumulated is None else accumulated + term
        averaged[key] = accumulated
    return averaged


def conditional_weights(num_examples: list[int]) -> list[float]:
    if not num_examples or any(count < 0 for count in num_examples):
        raise ValueError(f"Invalid example counts for FedAvg: {num_examples}")
    if sum(num_examples) <= 0:
        raise ValueError("FedAvg needs at least one trained example")
    return [float(count) for count in num_examples]


# --------------------------------------------------------------------------- #
# Adapter directories (PEFT-compatible, normalised keys)
# --------------------------------------------------------------------------- #
def _peft_version() -> str | None:
    try:
        from peft import __version__

        return __version__
    except Exception:  # pragma: no cover - peft is optional for the coordinator
        return None


def build_adapter_config(config: dict, base_model_name_or_path: str | None) -> dict:
    """A plain PEFT LORA config, so the averaged adapter loads on the PC base."""
    from softone_training import lora_kwargs

    args = lora_kwargs(config)
    return {
        "alpha": args["lora_alpha"],
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "bias": args["bias"],
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": args["lora_alpha"],
        "lora_dropout": args["lora_dropout"],
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": _peft_version(),
        "r": args["r"],
        "revision": None,
        "target_modules": args["target_modules"],
        "task_type": args["task_type"],
        "use_dora": args["use_dora"],
        "use_rslora": False,
    }


def assert_same_lora_config(reference: dict, candidate: dict, label: str) -> None:
    """Round configurations must match exactly or the average is meaningless."""
    for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "task_type", "bias",
                "use_dora"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(
                f"{label} LoRA config differs on {key}: "
                f"{candidate.get(key)!r} != {reference.get(key)!r}"
            )


def save_adapter(
    directory: Path,
    state: dict,
    config: dict,
    base_model_name_or_path: str | None,
    meta: dict | None = None,
) -> Path:
    """Write a normalised PEFT adapter directory; the caller owns overwrite checks."""
    stray = sorted(key for key in state if ".lora_" not in key)
    if stray:
        raise ValueError(f"save_adapter expects LoRA tensors only, got {stray[:4]}")
    if not state:
        raise ValueError("No LoRA tensors to save")
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700)
    tensors = {
        normalise_lora_key(key): value.detach().to("cpu").contiguous()
        for key, value in normalise_state(state).items()
    }
    save_file(tensors, str(directory / WEIGHT_FILE), metadata={"format": "pt"})
    payload = build_adapter_config(config, base_model_name_or_path)
    (directory / CONFIG_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    if meta is not None:
        (directory / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    return directory


def load_adapter(directory: Path) -> tuple[dict, dict]:
    """Return (normalised tensors, adapter_config.json) for an adapter directory."""
    from safetensors.torch import load_file

    directory = Path(directory)
    state = normalise_state(load_file(str(directory / WEIGHT_FILE)))
    config = json.loads((directory / CONFIG_FILE).read_text())
    return state, config


def read_meta(directory: Path) -> dict:
    path = Path(directory) / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"Adapter {directory} has no {META_FILE}")
    return json.loads(path.read_text())


def average_round(pc_dir: Path, mac_dir: Path, out_dir: Path, config: dict,
                  base_pc: str, index: int) -> None:
    """FedAvg one round's two client adapters into ``out_dir``, weighted by examples."""
    pc_state, pc_config = load_adapter(pc_dir)
    mac_state, mac_config = load_adapter(mac_dir)
    assert_same_lora_config(pc_config, mac_config, "mac adapter")
    weights = {"pc": read_meta(pc_dir)["num_examples"], "mac": read_meta(mac_dir)["num_examples"]}
    averaged = average_adapters([(pc_state, weights["pc"]), (mac_state, weights["mac"])])
    save_adapter(out_dir, averaged, config, base_pc,
                 {"round": index, "weights": weights, "keys": len(averaged)})


if __name__ == "__main__":
    # The coordinator runs in the app env (no torch); it calls this with the train venv.
    import sys

    _, pc, mac, out, config_path, base, round_index = sys.argv
    average_round(Path(pc), Path(mac), Path(out), json.loads(Path(config_path).read_text()),
                  base, int(round_index))
