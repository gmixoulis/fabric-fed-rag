#!/usr/bin/env python
"""PC-side FedAvg coordinator: two machines, one adapter, no GPU on this side.

This is plain python. It runs in the repo's ``uv`` environment, never imports
``torch``, never loads a model and never creates a CUDA context, so:

    uv run python training/fed/coordinator.py

Shards carry the plain catalog prompt text, so each client applies its own base
model's chat template and tokenizer. Each round the PC client runs under
``train-safe`` (finite systemd scope) and the Mac client runs under
``mac_run.sh`` (RSS/time watchdog); both must succeed or the whole run stops.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.umask(0o077)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fedavg  # noqa: E402
import softone_training as common  # noqa: E402

DEFAULT_CONFIG = _HERE / "config.json"
DEFAULT_EXPERIMENTS = Path.home() / ".local/share/softone-chat/experiments"
DEFAULT_TRAIN_SAFE = Path.home() / ".local/bin/train-safe"
DEFAULT_PC_PYTHON = Path.home() / ".local/share/softone-chat/train-venv/bin/python"
DEFAULT_MAC_HOST = "mac"
DEFAULT_MAC_ROOT = "~/opt/fedtrain"
DEFAULT_MAC_PYTHON = "~/opt/fedtrain/.venv/bin/python"
DEFAULT_DATA = _ROOT / "eval/softone_train_seed.jsonl"
DEFAULT_HOLDOUT = _ROOT / "eval/softone_holdout.jsonl"

CLIENT_FILES = (
    "client.py",
    "fedavg.py",
    "softone_training.py",
    "mac_run.sh",
)
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]
DRY_RUN_EXAMPLES = 4


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
def split_shard(items: list, weights, seed: int) -> list[list]:
    """Deterministic, complete, disjoint split into [PC, Mac] shards."""
    if len(items) < 2:
        raise ValueError("Need at least two examples to build two shards")
    shares = [float(weight) for weight in weights]
    if len(shares) != 2:
        raise ValueError(f"shard_weights must hold two values, got {weights!r}")
    if any(share < 0 for share in shares) or sum(shares) <= 0:
        raise ValueError(f"shard_weights must be non-negative and not all zero: {weights!r}")
    order = list(range(len(items)))
    random.Random(int(seed)).shuffle(order)
    pc_count = round(len(items) * shares[0] / sum(shares))
    pc_count = min(max(1, pc_count), len(items) - 1)
    first = [items[index] for index in order[:pc_count]]
    second = [items[index] for index in order[pc_count:]]
    return [first, second]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    path.chmod(0o600)
    return path


# --------------------------------------------------------------------------- #
# Command construction (kept pure so the tests can assert on it)
# --------------------------------------------------------------------------- #
def build_pc_command(
    *,
    train_safe: str,
    python: str,
    script: str,
    base: str,
    data: str,
    out: str,
    config: str,
    global_adapter: str | None = None,
    seed_offset: int = 0,
) -> list[str]:
    """PC client: always behind train-safe, so the scope limit cannot be skipped."""
    command = [
        train_safe, "--", python, script,
        "--base", base,
        "--data", data,
        "--out", out,
        "--device", "cuda",
        "--config", config,
        "--seed-offset", str(seed_offset),
    ]
    if global_adapter:
        command += ["--global-adapter", global_adapter]
    return command


def client_arguments(
    *,
    base: str,
    data: str,
    out: str,
    config: str,
    device: str,
    global_adapter: str | None,
    seed_offset: int,
) -> list[str]:
    arguments = [
        "--base", base,
        "--data", data,
        "--out", out,
        "--device", device,
        "--config", config,
        "--seed-offset", str(seed_offset),
    ]
    if global_adapter:
        arguments += ["--global-adapter", global_adapter]
    return arguments


def build_evaluate_command(
    *,
    train_safe: str,
    python: str,
    script: str,
    adapter: str,
    out: str,
    holdout: str,
    train_data: str,
) -> list[str]:
    """Post-run evaluation: the same train-safe wrapper as a PC round, no ssh."""
    return [
        train_safe, "--", python, script,
        "--adapter", adapter,
        "--out", out,
        "--holdout", holdout,
        "--train-data", train_data,
    ]


def evaluate_summary(report: dict) -> dict:
    """The two counts plus the case count, the part summary.json carries."""
    missing = [key for key in ("before_passed", "after_passed", "cases") if key not in report]
    if missing:
        raise RuntimeError(f"evaluation report is missing {missing}")
    return {key: report[key] for key in ("before_passed", "after_passed", "cases")}


def build_mac_command(
    *,
    host: str,
    remote_script: str,
    python: str,
    base: str,
    data: str,
    out: str,
    config: str,
    rss_limit_gb: float,
    timeout_s: int,
    global_adapter: str | None = None,
    seed_offset: int = 0,
) -> list[str]:
    """Mac client: one ssh hop, watchdog env first, the client arguments follow."""
    environment = [
        "env",
        f"FED_PYTHON={python}",
        f"FED_RSS_LIMIT_GB={rss_limit_gb}",
        f"FED_TIMEOUT_S={timeout_s}",
        "bash",
        remote_script,
    ]
    remote = shlex.join(
        environment
        + client_arguments(
            base=base, data=data, out=out, config=config, device="mps",
            global_adapter=global_adapter, seed_offset=seed_offset,
        )
    )
    return ["ssh", *SSH_OPTIONS, host, remote]


def expand_remote(path: str, mac_home: str) -> str:
    """Resolve a ``~`` path to an absolute Mac path so shlex quoting stays safe."""
    if path == "~":
        return mac_home
    if path.startswith("~/"):
        return mac_home.rstrip("/") + path[1:]
    return path


def read_mac_home(host: str) -> str:
    """One cheap ssh round trip; every other remote path is absolute from then on."""
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, host, 'printf %s "$HOME"'],
        capture_output=True, text=True, check=False,
    )
    home = result.stdout.strip()
    if result.returncode != 0 or not home.startswith("/"):
        raise RuntimeError(
            f"Cannot read $HOME from {host}: {result.stderr.strip() or result.returncode}"
        )
    return home


def build_ssh_command(host: str, words: list[str]) -> list[str]:
    return ["ssh", *SSH_OPTIONS, host, shlex.join(words)]


def build_mkdir_command(host: str, remote_dir: str) -> list[str]:
    return build_ssh_command(host, ["mkdir", "-p", remote_dir])


def build_scp_command(host: str, sources: list[str], remote_dir: str) -> list[str]:
    if not sources:
        raise ValueError("Nothing to copy to the Mac")
    return ["scp", "-q", "-r", *SSH_OPTIONS, *sources, f"{host}:{remote_dir}"]


def build_scp_fetch_command(host: str, remote_path: str, local_dir: str) -> list[str]:
    return ["scp", "-q", "-r", *SSH_OPTIONS, f"{host}:{remote_path}", local_dir]


def build_remote_kill_command(host: str, remote_round: str) -> list[str]:
    """Best effort: do not leave a 10 GiB client running on the Mac after an abort."""
    return build_ssh_command(host, ["pkill", "-f", f"{remote_round}/client.py"])


# --------------------------------------------------------------------------- #
# Process supervision
# --------------------------------------------------------------------------- #
def run_checked(command: list[str], log: Path, label: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {log}")


def kill_remote_round(host: str, remote_round: str, log: Path) -> None:
    command = build_remote_kill_command(host, remote_round)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)


def wait_for_clients(processes: dict[str, subprocess.Popen], timeout_s: float) -> dict[str, int]:
    """Wait for both clients; on timeout or failure kill the sibling, never retry."""
    pending = dict(processes)
    codes: dict[str, int] = {}
    deadline = time.monotonic() + timeout_s
    while pending:
        for name, process in list(pending.items()):
            code = process.poll()
            if code is not None:
                codes[name] = code
                del pending[name]
                if code != 0 and pending:
                    for sibling in pending.values():
                        sibling.kill()
                        sibling.wait()
                    pending.clear()
                    codes.update({other: -1 for other in processes if other not in codes})
        if not pending:
            break
        if time.monotonic() > deadline:
            for process in pending.values():
                process.kill()
            for process in pending.values():
                process.wait()
            raise RuntimeError(
                f"Round timed out after {timeout_s:.0f}s; killed {sorted(pending)}"
            )
        time.sleep(1)
    failed = {name: code for name, code in codes.items() if code != 0}
    if failed:
        raise RuntimeError(f"Clients failed ({failed}); aborting the federated run")
    return codes


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def read_seed_examples(path: Path) -> list[dict]:
    items = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(items) < 2:
        raise RuntimeError(f"Not enough examples in {path}")
    return items


def prepare_experiment(
    *, exp_dir: Path, data: Path, weights, seed: int, dry_run: bool
) -> list[list[dict]]:
    """Render (prompt, target) texts once, then split them into the two shards."""
    exp_dir.mkdir(parents=True, mode=0o700)
    exp_dir.chmod(0o700)
    texts = common.prepare_texts(_ROOT, read_seed_examples(data))
    if dry_run:
        texts = texts[:DRY_RUN_EXAMPLES]
    shards = split_shard(texts, weights, seed)
    for name, shard in zip(("pc", "mac"), shards, strict=True):
        write_jsonl(exp_dir / f"shard-{name}.jsonl", shard)
    (exp_dir / "shards.json").write_text(
        json.dumps([len(shard) for shard in shards]) + "\n"
    )
    return shards


# --------------------------------------------------------------------------- #
# One round
# --------------------------------------------------------------------------- #
def round_layout(exp_dir: Path, remote_root: str, index: int) -> dict:
    round_dir = exp_dir / f"round-{index}"
    remote_round = f"{remote_root}/runs/{exp_dir.name}/round-{index}"
    return {
        "dir": round_dir,
        "staging": round_dir / "remote",
        "pc_out": round_dir / "pc-adapter",
        "mac_out_local": round_dir / "mac-adapter",
        "mac_out_remote": f"{remote_round}/mac-adapter",
        "global": round_dir / "global",
        "remote_round": remote_round,
        "remote_script": f"{remote_round}/mac_run.sh",
        "remote_config": f"{remote_round}/config.json",
        "remote_shard": f"{remote_round}/shard.jsonl",
        "remote_global": f"{remote_round}/global",
    }


def stage_remote(round_info: dict, shard: Path, global_dir: Path | None,
                 config_path: Path) -> list[str]:
    """Local copy of exactly what the Mac will see, so a round is reproducible."""
    staging = round_info["staging"]
    staging.mkdir(parents=True, mode=0o700)
    for name in CLIENT_FILES:
        # softone_training.py is shared with the single-machine trainer in training/.
        source = _HERE / name if (_HERE / name).exists() else _HERE.parent / name
        shutil.copy2(source, staging / name)
    shutil.copy2(shard, staging / "shard.jsonl")
    # The effective config (dry-run overrides included), never the static one.
    shutil.copy2(config_path, staging / "config.json")
    sources = [str(staging / name) for name in (*CLIENT_FILES, "shard.jsonl", "config.json")]
    if global_dir is not None and Path(global_dir).is_dir():
        target = staging / "global"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(global_dir, target)
        sources.append(str(target))
    return sources


def push_round(host: str, round_info: dict, shard: Path, global_dir: Path | None,
               config_path: Path, log: Path) -> None:
    run_checked(build_mkdir_command(host, round_info["remote_round"]), log, "mac mkdir")
    sources = stage_remote(round_info, shard, global_dir, config_path)
    run_checked(build_scp_command(host, sources, round_info["remote_round"]), log, "scp push")


def client_meta(directory: Path) -> dict:
    meta = fedavg.read_meta(directory)
    for key in ("num_examples", "steps", "mean_loss_first", "mean_loss_last", "device",
                "seconds"):
        if key not in meta:
            raise RuntimeError(f"{directory}/meta.json is missing {key!r}")
    return meta


def round_summary(index: int, pc_meta: dict, mac_meta: dict, seconds: float) -> dict:
    return {
        "round": index,
        "seconds": round(seconds, 1),
        "weights": {"pc": pc_meta["num_examples"], "mac": mac_meta["num_examples"]},
        "pc": {key: pc_meta[key] for key in
               ("num_examples", "steps", "mean_loss_first", "mean_loss_last", "device",
                "seconds")},
        "mac": {key: mac_meta[key] for key in
                ("num_examples", "steps", "mean_loss_first", "mean_loss_last", "device",
                 "seconds")},
    }


def federated_average(pc_dir: Path, mac_dir: Path, out_dir: Path, config: dict,
                      base_pc: Path, index: int, pc_python: str = str(DEFAULT_PC_PYTHON)) -> dict:
    pc_meta, mac_meta = client_meta(pc_dir), client_meta(mac_dir)
    config_path = out_dir.parent / f"round-{index}-fedavg-config.json"
    config_path.write_text(json.dumps(config))
    # Averaging needs torch/safetensors, which live in the train venv, not the app env.
    subprocess.run([pc_python, str(_HERE / "fedavg.py"), str(pc_dir), str(mac_dir), str(out_dir),
                    str(config_path), str(base_pc), str(index)], check=True)
    return round_summary(index, pc_meta, mac_meta, 0.0)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--rounds", type=int, default=None, help="override config rounds")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 round, 1 local step, 2 examples per shard")
    parser.add_argument("--train-safe", default=str(DEFAULT_TRAIN_SAFE))
    parser.add_argument("--pc-python", default=str(DEFAULT_PC_PYTHON))
    parser.add_argument("--mac-host", default=DEFAULT_MAC_HOST)
    parser.add_argument("--mac-root", default=DEFAULT_MAC_ROOT)
    parser.add_argument("--mac-python", default=DEFAULT_MAC_PYTHON)
    parser.add_argument("--exp-name", default=None, help="default fed-<timestamp>")
    parser.add_argument("--evaluate", action="store_true",
                        help="score the final adapter before/after on the held-out set")
    parser.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = common.load_config(Path(args.config))
    rounds = int(args.rounds or config["rounds"])
    if args.dry_run:
        rounds = 1
        config["rounds"] = 1
        config["local_steps"] = 1
    base_pc = Path(config.get("base_pc", "~/.local/share/softone-chat/qwen35-4b-bnb"))
    base_pc = Path(base_pc).expanduser()
    base_mac = str(config.get("base_mac", "~/opt/fedtrain/models/qwen35-4b"))
    for path in (Path(args.train_safe), Path(args.pc_python), Path(args.data), base_pc):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.exp_name or f"fed-{stamp}" + ("-dryrun" if args.dry_run else "")
    exp_dir = Path(args.experiments).expanduser() / name
    if exp_dir.exists():
        raise SystemExit(f"Experiment already exists: {exp_dir}; refusing to overwrite")

    mac_home = read_mac_home(args.mac_host)
    mac_root = expand_remote(args.mac_root, mac_home)
    mac_python = expand_remote(args.mac_python, mac_home)
    base_mac = expand_remote(base_mac, mac_home)

    shards = prepare_experiment(
        exp_dir=exp_dir, data=Path(args.data),
        weights=config["shard_weights"], seed=int(config["seed"]), dry_run=args.dry_run,
    )
    config_copy = exp_dir / "config.json"
    config_copy.write_text(json.dumps(config, indent=2) + "\n")
    config_copy.chmod(0o600)
    print(json.dumps({"phase": "prepared", "experiment": str(exp_dir),
                      "shards": [len(shard) for shard in shards],
                      "rounds": rounds}), flush=True)

    logs = exp_dir / "logs"
    logs.mkdir(parents=True, mode=0o700)
    global_dir: Path | None = None
    summaries = []
    for index in range(1, rounds + 1):
        started = time.monotonic()
        layout = round_layout(exp_dir, mac_root, index)
        layout["dir"].mkdir(parents=True, mode=0o700)
        push_round(args.mac_host, layout, exp_dir / "shard-mac.jsonl", global_dir,
                   config_copy, logs / f"round-{index}-push.log")
        pc_command = build_pc_command(
            train_safe=args.train_safe, python=args.pc_python,
            script=str(_HERE / "client.py"), base=str(base_pc),
            data=str(exp_dir / "shard-pc.jsonl"), out=str(layout["pc_out"]),
            config=str(config_copy),
            global_adapter=None if global_dir is None else str(global_dir),
            seed_offset=index - 1,
        )
        mac_command = build_mac_command(
            host=args.mac_host, remote_script=layout["remote_script"],
            python=mac_python, base=base_mac, data=layout["remote_shard"],
            out=layout["mac_out_remote"], config=layout["remote_config"],
            rss_limit_gb=float(config["mac_rss_limit_gb"]),
            timeout_s=int(config["round_timeout_s"]),
            global_adapter=None if global_dir is None else layout["remote_global"],
            seed_offset=index - 1,
        )
        print(json.dumps({"phase": "round", "round": index, "pc": pc_command[0],
                          "mac": mac_command[:1]}), flush=True)
        with (layout["dir"] / "pc.log").open("w") as pc_log, \
                (layout["dir"] / "mac.log").open("w") as mac_log:
            pc = subprocess.Popen(pc_command, stdout=pc_log, stderr=subprocess.STDOUT)
            mac = subprocess.Popen(mac_command, stdout=mac_log, stderr=subprocess.STDOUT)
            try:
                wait_for_clients({"pc": pc, "mac": mac}, float(config["round_timeout_s"]))
            except RuntimeError as error:
                kill_remote_round(args.mac_host, layout["remote_round"],
                                  logs / f"round-{index}-abort.log")
                (exp_dir / "failure.json").write_text(
                    json.dumps({"round": index, "error": str(error),
                                "experiment": str(exp_dir)}, indent=2) + "\n"
                )
                print(json.dumps({"phase": "abort", "round": index, "error": str(error)}),
                      flush=True)
                return 1
        run_checked(
            build_scp_fetch_command(args.mac_host, layout["mac_out_remote"],
                                    str(layout["mac_out_local"])),
            logs / f"round-{index}-fetch.log", "scp fetch",
        )
        summary = federated_average(
            layout["pc_out"], layout["mac_out_local"], layout["global"], config, base_pc, index,
            args.pc_python,
        )
        summary["seconds"] = round(time.monotonic() - started, 1)
        summaries.append(summary)
        (layout["dir"] / "round.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"phase": "averaged", **summary}), flush=True)
        global_dir = layout["global"]

    final = exp_dir / "final-adapter"
    if global_dir is not None:
        shutil.copytree(global_dir, final)

    summary = {"experiment": str(exp_dir), "rounds": summaries, "final_adapter": str(final)}
    if args.evaluate:
        eval_out = exp_dir / "eval.json"
        command = build_evaluate_command(
            train_safe=args.train_safe, python=args.pc_python,
            script=str(_HERE / "evaluate_adapter.py"), adapter=str(final),
            out=str(eval_out), holdout=str(args.holdout), train_data=str(args.data),
        )
        print(json.dumps({"phase": "evaluate", "train_safe": command[0],
                          "out": str(eval_out)}), flush=True)
        try:
            run_checked(command, logs / "evaluate.log", "adapter evaluation")
            summary["evaluation"] = evaluate_summary(json.loads(eval_out.read_text()))
        except RuntimeError as error:
            # Keep the finished rounds on disk, then still fail the run.
            summary["evaluation"] = {"error": str(error)}
            (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"phase": "evaluate_failed", "error": str(error)}), flush=True)
            raise
        print(json.dumps({"phase": "evaluated", **summary["evaluation"]}), flush=True)

    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"phase": "done", "final_adapter": str(final)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"federated run aborted: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
