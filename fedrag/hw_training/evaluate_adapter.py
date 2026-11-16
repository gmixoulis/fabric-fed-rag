#!/usr/bin/env python
"""Score one adapter on the held-out set, before and after attaching it (PC only).

Run it behind ``train-safe`` from the training virtualenv::

    ~/.local/bin/train-safe <train-venv python> training/fed/evaluate_adapter.py \
        --adapter <experiment>/final-adapter --out <experiment>/eval.json

The base model is loaded exactly the way ``fed/client.py`` loads it for a PC
round (the pinned 4-bit bitsandbytes checkpoint), the held-out cases are scored
once with no adapter attached and once with the adapter attached, and both lists
plus the two counts are written to ``--out``.

Refusing to start outside a finite ``agent-training.scope`` or with less than
4 GiB of free GPU memory is deliberate: merely loading this model unbounded can
take the machine down, so the evaluation uses the same guards as training. The
report contains business questions and is therefore created mode 0600, and an
existing ``--out`` is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.umask(0o077)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (
    _HERE,
    _HERE.parent,
    _ROOT / "src",
    _ROOT / ".venv/lib/python3.12/site-packages",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import client  # noqa: E402
import softone_training as common  # noqa: E402

DEFAULT_HOLDOUT = _ROOT / "eval/softone_holdout.jsonl"
DEFAULT_TRAIN_DATA = _ROOT / "eval/softone_train_seed.jsonl"
DEFAULT_VERIFIED_CODES = _ROOT / "eval/softone_verified_codes.json"
DEFAULT_BASE = Path.home() / ".local/share/softone-chat/qwen35-4b-bnb"
DEFAULT_SNAPSHOT = Path.home() / ".local/share/softone-chat/snapshots/softone_reputation.sqlite"
REPORT_KEYS = ("before_passed", "after_passed", "cases")


def check_output(path: Path | str) -> None:
    """The report holds business questions: never overwrite one."""
    if Path(path).exists():
        raise RuntimeError(f"Output already exists: {path}; refusing to overwrite")


def read_verified_codes(path: Path | str) -> dict[tuple[str, str], set[str]]:
    """Read ``{"TABLE": {"COLUMN": [codes]}}`` as the guard's ``(TABLE, COLUMN)`` map.

    These are the codes a human taught through the memory store; the holdout's
    code cases are scored with exactly this mapping, so a model that invents a
    code is rejected even when its arithmetic would have matched.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    codes: dict[tuple[str, str], set[str]] = {}
    for table, columns in payload.items():
        for column, values in columns.items():
            key = (str(table).strip().upper(), str(column).strip().upper())
            codes[key] = {str(value).strip() for value in values}
    return codes


def assert_no_training_overlap(cases: list[dict], training: list[dict],
                               label: str = "held-out") -> None:
    """Fail loudly when an evaluation question is also a training question."""
    common.assert_no_question_overlap(cases, training, label)


def write_private_json(path: Path | str, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def count_passed(records: list[dict]) -> int:
    return sum(1 for record in records if record.get("passed"))


def evaluate_adapter(
    *,
    adapter: Path | str,
    out: Path | str,
    holdout: Path | str = DEFAULT_HOLDOUT,
    train_data: Path | str = DEFAULT_TRAIN_DATA,
    base: Path | str = DEFAULT_BASE,
    snapshot: Path | str = DEFAULT_SNAPSHOT,
    verified_codes: Path | str = DEFAULT_VERIFIED_CODES,
    max_new_tokens: int = 128,
) -> dict:
    """Guard, load the PC base, score before and after, write the report."""
    limits = common.assert_guarded_scope()
    print(json.dumps({"guard": "cgroup", **limits}), flush=True)
    free, total = common.assert_gpu_headroom()
    print(json.dumps({"gpu_free_gib": round(free / common.GIB, 2),
                      "gpu_total_gib": round(total / common.GIB, 2)}), flush=True)

    check_output(out)
    for path in (adapter, base, snapshot, holdout, train_data, verified_codes):
        if not Path(path).exists():
            raise RuntimeError(f"Missing required path: {path}")

    cases = common.load_cases(holdout)
    assert_no_training_overlap(cases, common.load_cases(train_data))
    codes = read_verified_codes(verified_codes)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(base), local_files_only=True, trust_remote_code=False
    )
    print(json.dumps({"phase": "loading", "base": str(base)}), flush=True)
    model = client.load_base_model(Path(base), "cuda", {})

    print(json.dumps({"phase": "before", "cases": len(cases)}), flush=True)
    before = common.evaluate_cases(
        model, tokenizer, cases, snapshot, "cuda",
        max_new_tokens=max_new_tokens, verified_codes=codes,
    )
    print(json.dumps({"phase": "before_done", "passed": count_passed(before)}), flush=True)

    print(json.dumps({"phase": "adapter", "path": str(adapter)}), flush=True)
    # Adapters are saved as PEFT directories by fedavg.save_adapter, so PEFT's own
    # loader attaches them; apply_global_adapter would need the round config too.
    model = PeftModel.from_pretrained(model, str(adapter))
    after = common.evaluate_cases(
        model, tokenizer, cases, snapshot, "cuda",
        max_new_tokens=max_new_tokens, verified_codes=codes,
    )

    report = {
        "adapter": str(adapter),
        "base": str(base),
        "holdout": str(holdout),
        "snapshot": str(snapshot),
        "before": before,
        "after": after,
        "before_passed": count_passed(before),
        "after_passed": count_passed(after),
        "cases": len(cases),
    }
    write_private_json(out, report)
    print(json.dumps({"phase": "done", **{key: report[key] for key in REPORT_KEYS}}),
          flush=True)
    return report


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="adapter directory to score")
    parser.add_argument("--out", required=True, help="report path (must not exist)")
    parser.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA),
                        help="training file the holdout must not overlap")
    parser.add_argument("--base", default=str(DEFAULT_BASE),
                        help="PC 4-bit base directory, as for a federated round")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--verified-codes", default=str(DEFAULT_VERIFIED_CODES))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_adapter(
        adapter=args.adapter,
        out=args.out,
        holdout=args.holdout,
        train_data=args.train_data,
        base=args.base,
        snapshot=args.snapshot,
        verified_codes=args.verified_codes,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"adapter evaluation refused: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
