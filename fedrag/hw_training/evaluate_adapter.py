#!/usr/bin/env python
"""Score one or more adapters on the held-out set, before and after attaching them.

Run it behind ``train-safe`` from the training virtualenv::

    ~/.local/bin/train-safe <train-venv python> training/fed/evaluate_adapter.py \
        --adapter <experiment>/final-adapter --out <experiment>/eval.json

The base model is loaded exactly the way ``fed/client.py`` loads it for a PC
round (the pinned 4-bit bitsandbytes checkpoint). The held-out cases are scored
once with no adapter attached, then once per ``--adapter`` in the order given,
and the reports (pass counts with a Wilson interval, teacher-forced held-out
loss, and the full records) are written to ``--out``.

Every score and every loss is computed against the production prompt: the taught
facts from ``eval/taught_facts.jsonl`` are passed as ``memories`` exactly as the
live pipeline does, so a held-out case that needs a taught type code is scored
with the same prompt the model will really receive.

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
DEFAULT_FACTS = _ROOT / "eval/taught_facts.jsonl"
DEFAULT_BASE = Path.home() / ".local/share/softone-chat/qwen35-4b-bnb"
DEFAULT_SNAPSHOT = Path.home() / ".local/share/softone-chat/snapshots/softone_reputation.sqlite"
DEFAULT_MAX_LENGTH = 2304  # the training max_length; 2048 left ~15 tokens spare
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


def heldout_case_losses(model, tokenizer, cases: list[dict], device: str, max_length: int,
                        facts: list[dict] | None) -> dict:
    """Teacher-forced loss of the gold answer per case id, built exactly like training.

    Each case is turned into the same ``{id, question, prompt, target}`` record
    ``prepare_texts`` produces (with the taught facts in the prompt), then
    ``client.encode_example`` and ``client.supervised_loss`` turn it into a loss
    comparable with the training loss. A case over ``max_length`` raises rather
    than being silently dropped.
    """
    import torch

    model.eval()
    losses: dict = {}
    with torch.no_grad():
        for case in cases:
            record = {
                "id": case["id"],
                "question": case["question"],
                "prompt": common.prompt_text(case["question"], facts=facts),
                "target": case["sql"],
            }
            batch = client.encode_example(tokenizer, record, max_length)
            losses[case["id"]] = float(client.supervised_loss(model, batch, device, torch).item())
    return losses


def mean_loss(case_losses: dict) -> float:
    """The mean of ``heldout_case_losses``; 0.0 for no cases."""
    if not case_losses:
        return 0.0
    values = list(case_losses.values())
    return float(sum(values) / len(values))


def heldout_loss(model, tokenizer, cases: list[dict], device: str, max_length: int,
                 facts: list[dict] | None) -> float:
    """Mean teacher-forced loss of the gold answer (see ``heldout_case_losses``)."""
    return mean_loss(heldout_case_losses(model, tokenizer, cases, device, max_length, facts))


def reuse_base(path: Path | str, cases: list[dict]) -> tuple[dict, list[dict]]:
    """The base summary and records of an earlier report, checked against ``cases``."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    before = report["before"]
    if [record["id"] for record in before] != [case["id"] for case in cases]:
        raise RuntimeError(f"{path} was scored on different holdout cases")
    return report["base"], before


def evaluate_adapter(
    *,
    adapters: list[Path | str],
    out: Path | str,
    holdout: Path | str = DEFAULT_HOLDOUT,
    train_data: Path | str = DEFAULT_TRAIN_DATA,
    base: Path | str = DEFAULT_BASE,
    snapshot: Path | str = DEFAULT_SNAPSHOT,
    verified_codes: Path | str = DEFAULT_VERIFIED_CODES,
    facts: Path | str = DEFAULT_FACTS,
    max_new_tokens: int = 128,
    max_length: int = DEFAULT_MAX_LENGTH,
    repair: bool = True,
    base_report: Path | str | None = None,
) -> dict:
    """Guard, load the PC base once, then score no-adapter and each adapter.

    ``base_report`` reuses the no-adapter score of an earlier report on the same
    holdout instead of spending a full pass re-scoring the unchanged base model.
    """
    limits = common.assert_guarded_scope()
    print(json.dumps({"guard": "cgroup", **limits}), flush=True)
    free, total = common.assert_gpu_headroom()
    print(json.dumps({"gpu_free_gib": round(free / common.GIB, 2),
                      "gpu_total_gib": round(total / common.GIB, 2)}), flush=True)

    check_output(out)
    adapter_paths = [Path(path) for path in adapters]
    for path in [*adapter_paths, Path(base), Path(snapshot), Path(holdout),
                 Path(train_data), Path(verified_codes), Path(facts)]:
        if not path.exists():
            raise RuntimeError(f"Missing required path: {path}")

    cases = common.load_cases(holdout)
    assert_no_training_overlap(cases, common.load_cases(train_data))
    codes = read_verified_codes(verified_codes)
    taught_facts = common.load_taught_facts(facts)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(base), local_files_only=True, trust_remote_code=False
    )
    print(json.dumps({"phase": "loading", "base": str(base)}), flush=True)
    model = client.load_base_model(Path(base), "cuda", {})

    if base_report is not None:
        base_summary, before = reuse_base(base_report, cases)
        base_loss = base_summary["loss"]
        print(json.dumps({"phase": "before_reused", "from": str(base_report)}), flush=True)
    else:
        print(json.dumps({"phase": "before", "cases": len(cases)}), flush=True)
        before = common.evaluate_cases(
            model, tokenizer, cases, snapshot, "cuda",
            max_new_tokens=max_new_tokens, verified_codes=codes, facts=taught_facts,
            repair=repair,
        )
        base_case_losses = heldout_case_losses(model, tokenizer, cases, "cuda",
                                               max_length, taught_facts)
        base_loss = mean_loss(base_case_losses)
        base_summary = common.summarise_records(before)
        base_summary["loss"] = base_loss
        base_summary["case_losses"] = base_case_losses
        base_summary["records"] = before
    print(json.dumps({"phase": "before_done", "passed": count_passed(before),
                      "loss": base_loss, "interval": base_summary["interval"]}), flush=True)

    adapter_records: list[dict] = []
    last = before
    for index, adapter in enumerate(adapter_paths):
        print(json.dumps({"phase": "adapter", "path": str(adapter)}), flush=True)
        # Adapters are saved as PEFT directories by fedavg.save_adapter, so PEFT's
        # own loader attaches them; apply_global_adapter would need the round
        # config too. The first is attached to the bare base model; every later
        # adapter is stacked onto the same model and switched to by name.
        if index == 0:
            model = PeftModel.from_pretrained(model, str(adapter))
        else:
            name = f"adapter-{index}"
            model.load_adapter(str(adapter), adapter_name=name)
            model.set_adapter(name)
        records = common.evaluate_cases(
            model, tokenizer, cases, snapshot, "cuda",
            max_new_tokens=max_new_tokens, verified_codes=codes, facts=taught_facts,
            repair=repair,
        )
        case_losses = heldout_case_losses(model, tokenizer, cases, "cuda", max_length,
                                          taught_facts)
        loss = mean_loss(case_losses)
        comparison = common.compare_records(before, records)
        summary = common.summarise_records(records)
        summary["loss"] = loss
        summary["case_losses"] = case_losses
        summary["gained"] = comparison["gained"]
        summary["lost"] = comparison["lost"]
        summary["records"] = records
        adapter_records.append({"adapter": str(adapter), **summary})
        last = records
        print(json.dumps({"phase": "adapter_done", "adapter": str(adapter),
                          "passed": count_passed(records), "loss": loss,
                          "interval": summary["interval"],
                          "gained": comparison["gained"],
                          "lost": comparison["lost"]}), flush=True)

    report = {
        "adapter": str(adapter_paths[-1]),
        "base": base_summary,
        "holdout": str(holdout),
        "snapshot": str(snapshot),
        "before": before,
        "after": last,
        "before_passed": count_passed(before),
        "after_passed": count_passed(last),
        "cases": len(cases),
        "adapters": adapter_records,
    }
    write_private_json(out, report)
    print(json.dumps({"phase": "done", **{key: report[key] for key in REPORT_KEYS}}),
          flush=True)
    return report


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", required=True,
                        help="adapter directory to score (repeatable, scored in order)")
    parser.add_argument("--out", required=True, help="report path (must not exist)")
    parser.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA),
                        help="training file the holdout must not overlap")
    parser.add_argument("--base", default=str(DEFAULT_BASE),
                        help="PC 4-bit base directory, as for a federated round")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--verified-codes", default=str(DEFAULT_VERIFIED_CODES))
    parser.add_argument("--facts", default=str(DEFAULT_FACTS),
                        help="taught facts passed to the prompt as memories")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="token budget for the held-out loss")
    parser.add_argument("--no-repair", dest="repair", action="store_false",
                        help="skip the repair pass (passed_repair) and its extra generations")
    parser.add_argument("--base-report", default=None,
                        help="reuse the no-adapter score from this earlier report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_adapter(
        adapters=args.adapter,
        out=args.out,
        holdout=args.holdout,
        train_data=args.train_data,
        base=args.base,
        snapshot=args.snapshot,
        verified_codes=args.verified_codes,
        facts=args.facts,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        repair=args.repair,
        base_report=args.base_report,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"adapter evaluation refused: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
