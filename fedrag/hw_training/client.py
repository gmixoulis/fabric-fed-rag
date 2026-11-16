#!/usr/bin/env python
"""Train one federated LoRA round on this machine (PC ``cuda`` or Mac ``mps``).

The coordinator prepares the shard (already chat-templated prompt + target SQL)
so this script needs only a tokenizer, the base model and the shared loss.  It
refuses to start without an enforced memory limit: a finite systemd scope on
CUDA, ``torch.mps.set_per_process_memory_fraction`` on MPS.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.umask(0o077)

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fedavg  # noqa: E402
import softone_training as common  # noqa: E402

DEFAULT_CONFIG = _HERE / "config.json"


# --------------------------------------------------------------------------- #
# Input validation and guards
# --------------------------------------------------------------------------- #
def encode_example(tokenizer, record: dict, max_length: int) -> dict:
    """Chat-template and tokenise one shard record with *this* machine's tokenizer."""
    prompt = common.render_chat(tokenizer, record["prompt"])
    return common.batch_for(
        tokenizer, prompt, record["target"] + tokenizer.eos_token, max_length
    )


def read_shard(path: Path) -> list[dict]:
    items = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("prompt") or not record.get("target"):
            raise RuntimeError(f"{path}:{number} needs both 'prompt' and 'target'")
        items.append(record)
    if not items:
        raise RuntimeError(f"Shard {path} is empty")
    return items


def check_output_dir(path: Path) -> None:
    if Path(path).exists():
        raise RuntimeError(f"Output already exists: {path}; refusing to overwrite")


def check_cuda_guard(cgroup_file: Path | None = None, cgroup_root: Path | None = None,
                     environ: dict[str, str] | None = None) -> dict[str, str]:
    """CUDA work must run inside the finite ``train-safe`` systemd scope."""
    limits = common.assert_guarded_scope(
        cgroup_file=common.CGROUP_FILE if cgroup_file is None else cgroup_file,
        cgroup_root=common.CGROUP_ROOT if cgroup_root is None else cgroup_root,
        environ=environ,
    )
    print(json.dumps({"guard": "cgroup", **limits}), flush=True)
    return limits


def check_mps_guard(config: dict, mps=None) -> dict:
    """MPS work must cap the allocator, or the 16 GiB machine swap-death spirals."""
    fraction = config.get("mps_memory_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
        raise RuntimeError(
            "mps_memory_fraction must be set in config.json; refusing unbounded MPS training"
        )
    if mps is None:
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend is unavailable on this machine")
        mps = torch.mps
    setter = getattr(mps, "set_per_process_memory_fraction", None)
    if setter is None:
        raise RuntimeError(
            "This torch.mps cannot enforce a memory fraction; refusing unbounded MPS training"
        )
    setter(float(fraction))
    info: dict[str, object] = {"fraction": float(fraction)}
    recommended = getattr(mps, "recommended_max_memory", None)
    if recommended is not None:
        info["recommended_bytes"] = int(recommended())
    print(json.dumps({"guard": "mps", **info}), flush=True)
    return info


# --------------------------------------------------------------------------- #
# Base model layout
# --------------------------------------------------------------------------- #
def is_multimodal_config(config_json: dict) -> bool:
    """True for the Qwen3.5 multimodal repo, whose text tower is nested."""
    return "text_config" in config_json and config_json.get("model_type") != "qwen3_5_text"


def text_checkpoint_path(base: Path) -> Path:
    base = Path(base)
    return base.parent / f"{base.name}-text"


def checkpoint_shards(base: Path, index: dict | None = None) -> list[Path]:
    if index is not None:
        names = sorted(set(index["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    missing = [name for name in names if not (Path(base) / name).exists()]
    if missing:
        raise RuntimeError(
            f"Base checkpoint {base} is incomplete: missing {missing}. Download the full "
            f"repository (huggingface-cli download Qwen/Qwen3.5-4B --local-dir {base}) "
            "before federated training; partial shards are only in .cache/huggingface."
        )
    return [Path(base) / name for name in names]


def ensure_text_checkpoint(base: Path) -> Path:
    """Return a text-only checkpoint directory, extracting it from the repo once."""
    base = Path(base)
    raw = json.loads((base / "config.json").read_text())
    if not is_multimodal_config(raw):
        return base
    target = text_checkpoint_path(base)
    if (target / "config.json").exists() and any(target.glob("*.safetensors")):
        print(json.dumps({"text_checkpoint": str(target), "reused": True}), flush=True)
        return target
    extract_text_checkpoint(base, target)
    print(json.dumps({"text_checkpoint": str(target), "reused": False}), flush=True)
    return target


def extract_text_checkpoint(base: Path, target: Path) -> None:
    """Copy only the language-model tensors out of the multimodal checkpoint.

    The multimodal repo holds ~9 GiB and its vision tower plus MTP head are dead
    weight for text-to-SQL.  Tensors are read one at a time so peak RSS stays at
    the text model size instead of the whole repository.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    shards = checkpoint_shards(base, index)
    config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    text_config = config.text_config
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(text_config, dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    expected = {name: parameter for name, parameter in model.named_parameters()}
    prefix = "model.language_model."
    filled: set[str] = set()
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if not key.startswith(prefix):
                        continue
                    name = "model." + key[len(prefix):]
                    parameter = expected.get(name)
                    if parameter is None:
                        raise RuntimeError(f"{name} from {shard.name} is not in the text model")
                    tensor = handle.get_tensor(key)
                    if tuple(tensor.shape) != tuple(parameter.shape):
                        raise RuntimeError(
                            f"Shape mismatch for {name}: {tuple(tensor.shape)} != "
                            f"{tuple(parameter.shape)}"
                        )
                    parameter.copy_(tensor)
                    filled.add(name)
    tied = {"lm_head.weight"}
    unfilled = sorted(set(expected) - filled - tied)
    if unfilled:
        raise RuntimeError(f"Text checkpoint left {len(unfilled)} tensors unfilled: {unfilled[:4]}")
    model.tie_weights()
    target.mkdir(parents=True, mode=0o700)
    model.save_pretrained(target, safe_serialization=True, max_shard_size="4GB")
    del model
    gc.collect()


def load_base_model(base: Path, device: str, config: dict):
    import torch
    from transformers import AutoModelForCausalLM

    if device == "cuda":
        from transformers import AutoConfig

        auto_config = AutoConfig.from_pretrained(
            base, local_files_only=True, trust_remote_code=False
        )
        if "quantization_config" not in auto_config.to_dict():
            raise RuntimeError(f"{base} is not the 4-bit bitsandbytes checkpoint")
        auto_config.quantization_config["bnb_4bit_compute_dtype"] = "float16"
        model = AutoModelForCausalLM.from_pretrained(
            base, config=auto_config, dtype=torch.float16, device_map={"": 0},
            local_files_only=True, trust_remote_code=False,
        )
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; refusing to train this shard on CPU")
        text_base = ensure_text_checkpoint(base)
        dtype = common.mps_torch_dtype(config.get("mps_dtype", "bfloat16"))
        try:
            # Load straight onto the device: a CPU copy plus an MPS copy would
            # double the resident size and blow the Mac RSS limit.
            model = AutoModelForCausalLM.from_pretrained(
                text_base, dtype=dtype, device_map={"": "mps"}, low_cpu_mem_usage=True,
                local_files_only=True, trust_remote_code=False,
            )
        except ImportError as error:
            raise RuntimeError(
                f"Loading the text checkpoint directly onto MPS needs accelerate: {error}"
            ) from error
    model.config.use_cache = False
    return model


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
def lora_state(model) -> dict:
    """Adapter tensors only.

    ``get_peft_model_state_dict`` also emits tied embeddings and ``lm_head`` for
    this model, which safetensors refuses because they share memory, and which
    must never be averaged across clients.
    """
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    return {name: tensor for name, tensor in state.items() if ".lora_" in name}


def lora_parameter_names(model) -> dict[str, tuple]:
    return {name: tuple(tensor.shape) for name, tensor in lora_state(model).items()}


def _values(shape: tuple[int, ...], token: str, bound: float) -> list[float]:
    """Deterministic uniform values, reproducible on both machines."""
    generator = random.Random(hashlib.sha256(token.encode()).hexdigest())
    return [generator.uniform(-bound, bound) for _ in range(int(_prod(shape)))]


def _prod(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


def initialise_lora_from_seed(model, seed: int) -> None:
    """Give both clients the same starting LoRA basis, keyed by tensor name.

    PEFT draws A from the global RNG, which differs per device and dtype; both
    machines then start from unrelated bases and the first FedAvg average would
    blend noise.  B stays zero, exactly like PEFT's own initialisation.
    """
    import torch

    with torch.no_grad():
        for name, tensor in sorted(model.named_parameters()):
            if ".lora_A." not in name and ".lora_B." not in name:
                continue
            if ".lora_A." in name:
                fan_in = tensor.shape[-1]
                bound = 1.0 / max(1, fan_in) ** 0.5
                values = _values(tuple(tensor.shape), f"{seed}:{name}", bound)
                tensor.copy_(torch.tensor(values, dtype=torch.float32).view(tensor.shape))
            else:
                tensor.zero_()


def build_peft_model(model, config: dict, seed: int):
    import torch
    from peft import get_peft_model

    torch.manual_seed(int(seed))
    wrapped = get_peft_model(model, common.build_lora_config(config))
    initialise_lora_from_seed(wrapped, int(seed))
    return wrapped


def apply_global_adapter(model, directory: Path, config: dict) -> None:
    """Load a FedAvg adapter, refusing silently dropped or reshaped tensors."""
    from peft import set_peft_model_state_dict

    state, adapter_config = fedavg.load_adapter(directory)
    fedavg.assert_same_lora_config(
        fedavg.build_adapter_config(config, None), adapter_config, "global adapter"
    )
    expected = lora_parameter_names(model)
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"Global adapter keys do not match this model: missing={missing[:4]} extra={extra[:4]}"
        )
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape:
            raise RuntimeError(
                f"Global adapter shape mismatch for {name}: {tuple(state[name].shape)} != {shape}"
            )
    set_peft_model_state_dict(model, state)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def supervised_loss(model, batch: dict, device: str, torch_module):
    torch = torch_module
    input_ids = torch.tensor([batch["ids"]], dtype=torch.long, device=device)
    size = int(batch["target_size"])
    base = model.get_base_model()

    def compute():
        # The full vocabulary head cannot be cast at once on a 6 GiB GPU.
        hidden = base.model(input_ids=input_ids, use_cache=False).last_hidden_state
        return common.chunked_lm_loss(
            hidden[:, -size - 1:-1, :], base.lm_head.weight, input_ids[:, -size:]
        )

    if device == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            return compute()
    return compute()


def train_local(model, tokenizer, examples: list[dict], config: dict, device: str,
                seed_offset: int = 0) -> list[float]:
    import torch

    steps = int(config["local_steps"])
    accumulation = max(1, min(int(config["grad_accumulation"]), steps))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["lr"]))
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    generator = random.Random(int(config["seed"]) + int(seed_offset))
    order: list[int] = []
    while len(order) < steps:
        block = list(range(len(examples)))
        generator.shuffle(block)
        order.extend(block)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses = []
    for index in range(steps):
        item = examples[order[index]]
        batch = encode_example(tokenizer, item, int(config["max_length"]))
        loss = supervised_loss(model, batch, device, torch) / accumulation
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaled = scaler.scale(loss) if scaler is not None else loss
        scaled.backward()
        losses.append(loss.item() * accumulation)
        finished = (index + 1) % accumulation == 0 or index + 1 == steps
        if finished:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"step": index + 1, "loss": round(losses[-1], 4),
                          "example": item.get("id")}), flush=True)
    return losses


def summarise_losses(losses: list[float]) -> tuple[float, float]:
    if not losses:
        raise RuntimeError("No local steps ran")
    window = max(1, min(3, len(losses)))
    return round(sum(losses[:window]) / window, 4), round(sum(losses[-window:]) / window, 4)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base model directory")
    parser.add_argument("--data", required=True, help="this client's text shard (jsonl)")
    parser.add_argument("--global-adapter", default=None, help="FedAvg adapter directory")
    parser.add_argument("--out", required=True, help="adapter output directory (must not exist)")
    parser.add_argument("--device", required=True, choices=common.EXPECTED_DEVICES)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="round index, so each round shuffles differently")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = common.load_config(Path(args.config))
    base = Path(args.base).expanduser()
    out = Path(args.out).expanduser()
    check_output_dir(out)
    examples = read_shard(Path(args.data).expanduser())
    print(json.dumps({"phase": "shard", "examples": len(examples), "device": args.device,
                      "base": str(base)}), flush=True)

    if args.device == "cuda":
        check_cuda_guard()
        free, total = common.assert_gpu_headroom()
        print(json.dumps({"gpu_free_gib": round(free / common.GIB, 2),
                          "gpu_total_gib": round(total / common.GIB, 2)}), flush=True)
    else:
        check_mps_guard(config)

    import torch
    from transformers import AutoTokenizer

    started = time.monotonic()
    torch.manual_seed(int(config["seed"]) + int(args.seed_offset))
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False
    )
    print(json.dumps({"phase": "loading", "base": str(base)}), flush=True)
    model = load_base_model(base, args.device, config)
    model = build_peft_model(model, config, int(config["seed"]))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    if args.global_adapter:
        apply_global_adapter(model, Path(args.global_adapter).expanduser(), config)
        print(json.dumps({"phase": "global_adapter",
                          "path": str(args.global_adapter)}), flush=True)

    losses = train_local(model, tokenizer, examples, config, args.device, args.seed_offset)
    first, last = summarise_losses(losses)
    seconds = round(time.monotonic() - started, 2)
    meta = {
        "num_examples": len(examples),
        "steps": int(config["local_steps"]),
        "mean_loss_first": first,
        "mean_loss_last": last,
        "device": args.device,
        "seconds": seconds,
        "losses": losses,
        "seed_offset": int(args.seed_offset),
        "base": str(base),
        "torch": torch.__version__,
        "global_adapter": None if not args.global_adapter else str(args.global_adapter),
    }
    fedavg.save_adapter(out, lora_state(model), config, str(base), meta)
    print(json.dumps({"phase": "saved", "out": str(out), "seconds": seconds,
                      "mean_loss_first": first, "mean_loss_last": last}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"fed client refused: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
