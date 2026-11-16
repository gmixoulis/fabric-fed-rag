"""Pieces shared by the single-machine trainer and the federated clients.

``training/train_softone_qwen.py`` (PC only), ``training/fed/client.py`` (PC and
Mac) and ``training/fed/coordinator.py`` (PC only) all use this module so the
chunked cross entropy, the safety checks and the (prompt, target) text building
exist exactly once.

Torch, transformers, peft and ``softone_rag`` are imported lazily: the
coordinator copies this file to the Mac, where only torch/transformers/peft
exist and no weights are loaded at import time.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

GIB = 1024**3
CGROUP_FILE = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
SCOPE_MARKER = "agent-training.scope"
MIN_FREE_GPU_BYTES = 4 * GIB
GRAD_ACCUMULATION = 4
EXPECTED_DEVICES = ("cuda", "mps")

REQUIRED_CONFIG_KEYS = (
    "rounds",
    "local_steps",
    "lr",
    "lora",
    "max_length",
    "seed",
    "mps_memory_fraction",
    "mac_rss_limit_gb",
    "round_timeout_s",
)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def scope_memory_limits(cgroup_text: str, cgroup_root: Path = CGROUP_ROOT) -> dict[str, str]:
    """Read memory.max/memory.swap.max for the cgroup listed in ``cgroup_text``."""
    _, _, relative = cgroup_text.strip().partition("::")
    if not relative:
        raise RuntimeError(f"Unparsable cgroup entry: {cgroup_text!r}")
    scope = Path(cgroup_root) / relative.lstrip("/")
    memory = (scope / "memory.max").read_text().strip()
    swap = (scope / "memory.swap.max").read_text().strip()
    return {"cgroup": relative, "memory_max": memory, "memory_swap_max": swap}


def assert_guarded_scope(
    cgroup_file: Path = CGROUP_FILE,
    cgroup_root: Path = CGROUP_ROOT,
    environ: dict[str, str] | None = None,
    require_marker: bool = True,
) -> dict[str, str]:
    """Refuse to train unless this process runs inside a finite memory cgroup.

    ``train-safe`` sets ``AGENT_TRAINING_GUARDED=1`` and runs the worker in
    ``agent-training.scope``; a CUDA client that would escape those limits must
    fail here rather than take the whole machine down.
    """
    environment = os.environ if environ is None else environ
    if environment.get("AGENT_TRAINING_GUARDED") != "1":
        raise RuntimeError(
            "Use train-safe; AGENT_TRAINING_GUARDED is not set for this worker"
        )
    text = Path(cgroup_file).read_text()
    limits = scope_memory_limits(text, cgroup_root)
    if require_marker and SCOPE_MARKER not in limits["cgroup"]:
        raise RuntimeError(
            f"Worker is not in {SCOPE_MARKER} (found {limits['cgroup']!r})"
        )
    if not limits["cgroup"].endswith(".scope"):
        raise RuntimeError(f"Worker cgroup is not a scope: {limits['cgroup']!r}")
    if limits["memory_max"] == "max" or limits["memory_swap_max"] == "max":
        raise RuntimeError(
            "Training cgroup has unbounded memory: "
            f"memory.max={limits['memory_max']} memory.swap.max={limits['memory_swap_max']}"
        )
    return limits


def assert_gpu_headroom(minimum_bytes: int = MIN_FREE_GPU_BYTES) -> tuple[int, int]:
    """Require ``minimum_bytes`` of free GPU memory before touching the model."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing unbounded CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < minimum_bytes:
        raise RuntimeError(
            f"Less than {minimum_bytes / GIB:.0f} GiB of GPU memory available "
            f"({free / GIB:.2f}/{total / GIB:.2f} GiB); no training started"
        )
    return free, total


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def chunked_lm_loss(hidden, weights, targets, chunk_size=8192):
    """Exact cross entropy, checkpointed in vocabulary chunks to avoid a full head cast."""
    import torch
    from torch.utils.checkpoint import checkpoint

    log_normalizer = None
    target_logit = torch.zeros_like(targets, dtype=torch.float32)
    for start in range(0, weights.shape[0], chunk_size):
        end = min(start + chunk_size, weights.shape[0])

        def chunk(value, first=start, last=end, last_exclusive=end):
            logits = torch.nn.functional.linear(value, weights[first:last]).float()
            indices = (targets - first).clamp(0, last - first - 1)
            selected = logits.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            selected = torch.where(
                (targets >= first) & (targets < last_exclusive), selected, 0.0
            )
            return logits.logsumexp(dim=-1), selected

        normalizer, selected = checkpoint(chunk, hidden, use_reentrant=False)
        log_normalizer = (
            normalizer if log_normalizer is None else torch.logaddexp(log_normalizer, normalizer)
        )
        target_logit = target_logit + selected
    return (log_normalizer - target_logit).mean()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_seed_examples(root: Path) -> list[dict]:
    """Read the (question, sql) seed pairs; order is the file order."""
    text = (Path(root) / "eval/softone_train_seed.jsonl").read_text()
    examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(examples) < 2:
        raise RuntimeError("Not enough seed examples to train and split")
    return examples


def prompt_text(question: str) -> str:
    """The production prompt text for one question (needs the repo venv)."""
    from softone_rag import graph, real_catalog

    return graph.build_prompt(question, list(real_catalog.TABLES), data_catalog=real_catalog)


def render_chat(tokenizer, prompt: str) -> str:
    """Wrap a prompt in the model's own chat template, without thinking traces."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def chat_prompt(tokenizer, question: str) -> str:
    """Render the production prompt for one question with the Qwen chat template."""
    return render_chat(tokenizer, prompt_text(question))


def prepare_texts(root: Path, examples: list[dict] | None = None) -> list[dict]:
    """(prompt, target) texts, so a shard needs only a tokenizer to train.

    The prompt is the plain catalog prompt, not chat-templated: each client
    applies its own base model's template, so a shard carries no token ids and no
    template assumptions from the coordinating machine.
    """
    items = load_seed_examples(root) if examples is None else examples
    return [
        {
            "id": index,
            "question": example["question"],
            "prompt": prompt_text(example["question"]),
            "target": example["sql"],
        }
        for index, example in enumerate(items)
    ]


def batch_for(tokenizer, prompt: str, target: str, max_length: int) -> dict:
    """Tokenise one example and locate the supervised suffix; no tensors involved."""
    prefix_size = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(prompt + target, add_special_tokens=False)["input_ids"]
    target_size = len(ids) - prefix_size
    if target_size < 1:
        raise RuntimeError("Invalid example: no supervised target tokens")
    if len(ids) > max_length:
        raise RuntimeError(f"Example is {len(ids)} tokens, over max_length={max_length}")
    return {"ids": ids, "prefix_size": prefix_size, "target_size": target_size}


# --------------------------------------------------------------------------- #
# Held-out evaluation
# --------------------------------------------------------------------------- #
def normalised_question(text: str) -> str:
    """Whitespace-collapsed, case-folded question text, for overlap checks."""
    return " ".join(str(text or "").split()).casefold()


def load_cases(path: Path | str) -> list[dict]:
    """Read ``{id, question, sql}`` cases; ids must be unique when they are given.

    The training file carries no ids, so a case without one gets ``case-<line>``
    as a stable label for the evaluation report.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such JSONL file: {path}")
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("question", "sql"):
            if not str(case.get(key) or "").strip():
                raise ValueError(f"{path}:{number} has no {key}")
        case.setdefault("id", f"case-{number}")
        if not str(case["id"]).strip():
            raise ValueError(f"{path}:{number} has an empty id")
        cases.append(dict(case))
    if not cases:
        raise ValueError(f"{path} holds no cases")
    ids = [case["id"] for case in cases]
    duplicated = sorted({value for value in ids if ids.count(value) > 1})
    if duplicated:
        raise ValueError(f"{path} repeats ids: {duplicated}")
    return cases


def assert_no_question_overlap(cases: list[dict], training: list[dict],
                               label: str = "held-out") -> None:
    """Fail loudly when an evaluation question is also a training question."""
    trained = {normalised_question(record["question"]) for record in training}
    leaked = sorted({normalised_question(case["question"]) for case in cases} & trained)
    if leaked:
        raise RuntimeError(
            f"{label} questions also appear in the training data (leakage): "
            + "; ".join(leaked[:3])
        )


def gold_is_ordered(sql: str) -> bool:
    """True when the gold SQL orders its rows, so the comparison must keep order."""
    return bool(re.search(r"\border\s+by\b", str(sql or ""), re.IGNORECASE))


def generate_text(model, tokenizer, prompt: str, device: str,
                  max_new_tokens: int = 128) -> str:
    """Greedy continuation after ``prompt``. The only torch-dependent step."""
    import torch

    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **tokens, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, tokens.input_ids.shape[1]:],
                            skip_special_tokens=True)


def case_passes(text: str, case: dict, snapshot, verified_codes=None) -> bool:
    """Did a reply answer the case: guarded SQL whose rows match the gold rows.

    A case with no gold SQL expects a refusal, which is how the single-machine
    trainer's ``CANNOT_ANSWER`` cases are expressed. Otherwise the reply must
    contain SQL that the guard accepts with the catalog and the taught codes, and
    that SQL must return the same rows as the gold SQL.
    """
    from softone_rag import db, graph, guard, real_catalog
    from softone_rag import evaluate as evaluator
    from softone_rag.llm import CANNOT_ANSWER

    if not case.get("sql"):
        return CANNOT_ANSWER in text
    sql = graph.extract_sql(text)
    if sql is None:
        return False
    checked = guard.validate_sql(sql, real_catalog.ALLOWED_TABLES,
                                 real_catalog.UNVERIFIED_CODES, verified_codes or {})
    if not checked.ok:
        return False
    try:
        _, actual = db.query(snapshot, checked.sql)
        _, expected = db.query(snapshot, case["sql"])
    except db.DatabaseError:
        return False
    if not actual and not expected:
        # Two empty results are not a match; the SQL itself must be equivalent.
        return evaluator.sql_equivalent(case["sql"], checked.sql)
    return evaluator.results_equal(actual, expected, gold_is_ordered(case["sql"]))


def evaluate_cases(model, tokenizer, cases: list[dict], snapshot, device: str,
                   max_new_tokens: int = 128,
                   verified_codes=None) -> list[dict]:
    """Score every held-out case at one point in time; one record per case.

    ``verified_codes`` is the same ``(TABLE, COLUMN) -> {values}`` mapping the live
    pipeline gets from ``MemoryStore.verified_codes``; without it the guard
    rejects every comparison against an unverified code column, which is the
    honest outcome when nobody has taught those codes.
    """
    import time

    device = str(device)
    model.eval()
    results = []
    for case in cases:
        started = time.monotonic()
        record = {"id": case["id"], "passed": False, "seconds": 0.0}
        try:
            prompt = chat_prompt(tokenizer, case["question"])
            text = generate_text(model, tokenizer, prompt, device, max_new_tokens)
            record["passed"] = case_passes(text, case, snapshot, verified_codes)
        except Exception as error:  # one broken case must not end the evaluation
            record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = round(time.monotonic() - started, 2)
        results.append(record)
    return results


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    """Read config.json and reject values that would silently weaken the run."""
    config = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"config.json is missing {missing}")
    lora = config["lora"]
    for key in ("r", "alpha", "dropout", "target_modules"):
        if key not in lora:
            raise ValueError(f"config lora is missing {key!r}")
    checks = [
        (int(config["rounds"]) >= 1, "rounds must be >= 1"),
        (int(config["local_steps"]) >= 1, "local_steps must be >= 1"),
        (float(config["lr"]) > 0, "lr must be > 0"),
        (int(lora["r"]) >= 1, "lora.r must be >= 1"),
        (int(lora["alpha"]) >= 1, "lora.alpha must be >= 1"),
        (0.0 <= float(lora["dropout"]) < 1.0, "lora.dropout must be in [0, 1)"),
        (len(lora["target_modules"]) >= 1, "lora.target_modules must not be empty"),
        (int(config["max_length"]) >= 64, "max_length must be >= 64"),
        (0.0 < float(config["mps_memory_fraction"]) <= 1.0,
         "mps_memory_fraction must be in (0, 1]"),
        (float(config["mac_rss_limit_gb"]) > 0, "mac_rss_limit_gb must be > 0"),
        (int(config["round_timeout_s"]) > 0, "round_timeout_s must be > 0"),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise ValueError("invalid config: " + "; ".join(failed))
    config.setdefault("grad_accumulation", GRAD_ACCUMULATION)
    config.setdefault("mps_dtype", "bfloat16")
    config.setdefault("shard_weights", [0.5, 0.5])
    return config


def lora_kwargs(config: dict) -> dict:
    """The exact PEFT constructor arguments; identical on both clients."""
    lora = config["lora"]
    return {
        "r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
        "target_modules": list(lora["target_modules"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def build_lora_config(config: dict):
    from peft import LoraConfig

    return LoraConfig(**lora_kwargs(config))


def mps_torch_dtype(name: str):
    import torch

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"Unsupported mps_dtype {name!r}; use one of {sorted(mapping)}") from None
