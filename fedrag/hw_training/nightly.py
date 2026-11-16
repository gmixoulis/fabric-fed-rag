#!/usr/bin/env python
"""One nightly cycle: both machines train, both machines are measured, one report.

Plain python in the **app** environment (the one ``softone-rag`` is installed in).
It never imports ``torch`` and never loads a model; every piece of heavy work runs
in a subprocess through an injected runner:

- ``softone-rag export-training`` writes the dated training file (app env).
- ``training/fed/coordinator.py`` drives the federated rounds: the PC client under
  ``train-safe``, the Mac client over ``ssh mac`` (it starts both by itself).
- ``softone-rag eval --real`` scores the production pipeline against a snapshot.

The two machines are stopped first because training on either side needs the GPU
free, and they are restored in ``finally`` in the reverse order: the Mac RPC
worker, then ``llama-split``, then the two chat services. Nothing is deleted, ever.

State lives in ``nightly-state.json`` next to the lock, so a run can tell whether
one more training round is worth it.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import coordinator  # noqa: E402  (shares the ssh options and the Mac host name)

DEFAULT_HOME = Path.home() / ".local/share/softone-chat"
DEFAULT_APP_PYTHON = sys.executable
DEFAULT_MIN_NEW = 20
DEFAULT_MAC_HOST = coordinator.DEFAULT_MAC_HOST
RUN_PREFIX = "nightly"
DATE_FORMAT = "%Y%m%d"

# The chat services come down first, then the shared llama-server, then the Mac
# worker; restoring is the exact reverse (see ``restore_services``).
PC_UNITS = ("softone-chat", "softone-chat-public", "llama-split")
MAC_RPC_LABEL = "local.llama.rpc"
MAC_RPC_PLIST = f"~/Library/LaunchAgents/{MAC_RPC_LABEL}.plist"

LLAMA_HEALTH_URL = "http://127.0.0.1:8095/health"
HEALTH_TIMEOUT_S = 900.0
HEALTH_INTERVAL_S = 5.0

# ``eval/real_cases.jsonl`` carries absolute ``expected_numbers`` (14 customers,
# 1523 items, ...) taken from one restored backup, so the same case file is only
# meaningful on that snapshot. Evaluated by default: ``softone_reputation``, which
# is exactly the database ``eval --real`` defaults to and the numbers match.
DEFAULT_EVAL_DATABASES = ("softone_reputation",)
EVAL_SCOPE_NOTE = ("the expected numbers in eval/real_cases.jsonl only match "
                   "softone_reputation")

STATE_KEYS = ("last_trained_confirmed", "best_adapter", "best_after_passed")


# --------------------------------------------------------------------------- #
# Command runner (the only thing that touches the outside world)
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    returncode: int
    output: str = ""


class SubprocessRunner:
    """Runs one command, streaming its output to a private log file."""

    def run(self, argv, *, log: Path | None = None, cwd=None, capture: bool = False,
            timeout: float | None = None) -> Result:
        argv = [str(part) for part in argv]
        if capture or log is None:
            done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                  timeout=timeout, check=False)
            output = (done.stdout or "") + (done.stderr or "")
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
                write_private(log, output)
            return Result(done.returncode, output)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            done = subprocess.run(argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        return Result(done.returncode, "")


def write_private(path: Path | str, payload: str) -> Path:
    """Write a file only its owner can read; ``os.umask`` is not enough to rely on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.chmod(path, 0o600)
    return path


# --------------------------------------------------------------------------- #
# Single instance
# --------------------------------------------------------------------------- #
def acquire_lock(path: Path):
    """Exclusive non-blocking lock; ``None`` when another nightly run holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


# --------------------------------------------------------------------------- #
# Export and the training decision
# --------------------------------------------------------------------------- #
def export_command(app_python: str, out: Path) -> list[str]:
    return [app_python, "-m", "softone_rag", "export-training", str(out)]


def parse_counts(output: str, keys) -> dict[str, int]:
    """The ``  key: number`` lines the export CLI prints, keyed by COUNT_KEYS."""
    counts: dict[str, int] = {}
    for key in keys:
        match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", output, re.MULTILINE)
        if match:
            counts[key] = int(match.group(1))
    return counts


def export_examples(runner: SubprocessRunner, *, app_python: str, out: Path,
                    log: Path) -> dict[str, int]:
    """Run ``export-training`` into a new dated file and return the printed counts."""
    from softone_rag import training_export

    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = runner.run(export_command(app_python, out), log=log, cwd=_ROOT, capture=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"export-training failed (exit {result.returncode}); see {log}"
        )
    if not out.is_file():
        raise RuntimeError(f"export-training wrote no {out}")
    os.chmod(out, 0o600)
    counts = parse_counts(result.output, training_export.COUNT_KEYS)
    missing = [key for key in training_export.COUNT_KEYS if key not in counts]
    if missing:
        raise RuntimeError(f"export-training printed no counts for {missing}")
    return counts


@dataclass
class State:
    """What the previous nightly run left behind, so a run knows if it is worth it."""

    last_trained_confirmed: int = 0
    best_adapter: str | None = None
    best_after_passed: int | None = None
    last_run: str | None = None

    @classmethod
    def load(cls, path: Path) -> State:
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise RuntimeError(f"{path} is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} must hold a JSON object")
        best = payload.get("best_after_passed")
        return cls(
            last_trained_confirmed=int(payload.get("last_trained_confirmed") or 0),
            best_adapter=payload.get("best_adapter"),
            best_after_passed=None if best is None else int(best),
            last_run=payload.get("last_run"),
        )

    def save(self, path: Path) -> Path:
        return write_private(path, json.dumps({
            "last_trained_confirmed": self.last_trained_confirmed,
            "best_adapter": self.best_adapter,
            "best_after_passed": self.best_after_passed,
            "last_run": self.last_run,
        }, indent=2) + "\n")


def decide(*, confirmed: int, state: State, min_new: int, force: bool) -> tuple[bool, str]:
    """Train only when the human-confirmed set grew enough since the last run."""
    if force:
        return True, "forced with --force-train"
    new_examples = confirmed - state.last_trained_confirmed
    if new_examples >= min_new:
        return True, (
            f"{new_examples} new confirmed example(s) since the last trained run "
            f"({state.last_trained_confirmed}); at least {min_new} required"
        )
    if new_examples <= 0:
        return False, (
            f"no new confirmed examples since the last trained run "
            f"({state.last_trained_confirmed}); {min_new} required"
        )
    return False, (
        f"only {new_examples} new confirmed example(s) since the last trained run "
        f"({state.last_trained_confirmed}); {min_new} required"
    )


# --------------------------------------------------------------------------- #
# Services on both machines
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Service:
    name: str
    stop: list[str]
    start: list[str]


def services(mac_host: str) -> list[Service]:
    """Stop order. ``restore_services`` walks this list backwards."""
    ordered = [
        Service(unit,
                ["systemctl", "--user", "stop", f"{unit}.service"],
                ["systemctl", "--user", "start", f"{unit}.service"])
        for unit in PC_UNITS
    ]
    ordered.append(Service(
        "mac-rpc",
        # ``$(id -u)`` must reach the Mac's shell unconverted: the remote uid is not
        # this host's uid, so the words are joined with plain spaces, not shlex.
        ["ssh", *coordinator.SSH_OPTIONS, mac_host,
         f"launchctl bootout gui/$(id -u)/{MAC_RPC_LABEL}"],
        ["ssh", *coordinator.SSH_OPTIONS, mac_host,
         f"launchctl bootstrap gui/$(id -u) {MAC_RPC_PLIST}"],
    ))
    return ordered


def stop_services(runner: SubprocessRunner, ordered: list[Service], logs: Path) -> list[str]:
    """Free GPU memory on both machines; a failed stop aborts before training."""
    stopped = []
    for service in ordered:
        result = runner.run(service.stop, log=logs / f"stop-{service.name}.log")
        if result.returncode != 0:
            raise RuntimeError(
                f"could not stop {service.name} (exit {result.returncode}); GPU memory "
                f"is not free, so training was not started"
            )
        stopped.append(service.name)
    return stopped


def restore_services(runner: SubprocessRunner, ordered: list[Service],
                     logs: Path) -> list[str]:
    """Start everything again, Mac RPC first, and report what did not come back."""
    failed = []
    for service in reversed(ordered):
        try:
            result = runner.run(service.start, log=logs / f"start-{service.name}.log")
        except (OSError, subprocess.SubprocessError):
            failed.append(service.name)
            continue
        if result.returncode != 0:
            failed.append(service.name)
    return failed


def llama_health(url: str = LLAMA_HEALTH_URL, timeout: float = 5.0) -> bool:
    """llama-server answers 200 on ``/health`` once the model is resident."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def wait_for_llama_health(*, probe=llama_health, timeout_s: float = HEALTH_TIMEOUT_S,
                          interval_s: float = HEALTH_INTERVAL_S, clock=time.monotonic,
                          sleep=time.sleep) -> dict:
    started = clock()
    attempts = 0
    while True:
        attempts += 1
        if probe():
            return {"checked": True, "healthy": True, "attempts": attempts,
                    "seconds": round(clock() - started, 1)}
        if clock() - started >= timeout_s:
            return {"checked": True, "healthy": False, "attempts": attempts,
                    "seconds": round(clock() - started, 1),
                    "error": f"no answer from {LLAMA_HEALTH_URL} after {timeout_s:.0f}s"}
        sleep(interval_s)


# --------------------------------------------------------------------------- #
# Federated training and promotion
# --------------------------------------------------------------------------- #
def coordinator_command(*, app_python: str, data: Path, run_dir: Path,
                        exp_name: str) -> list[str]:
    """``--experiments`` is the dated run directory, so the coordinator's own
    experiment subdirectory never collides with the exported ``train.jsonl``."""
    return [
        app_python, str(_HERE / "coordinator.py"),
        "--data", str(data),
        "--evaluate",
        "--experiments", str(run_dir),
        "--exp-name", exp_name,
    ]


def train(runner: SubprocessRunner, *, app_python: str, data: Path, run_dir: Path,
          exp_name: str, log: Path) -> dict:
    """Run the coordinator (PC + Mac) and read the evaluation counts it wrote."""
    result = runner.run(coordinator_command(app_python=app_python, data=data,
                                            run_dir=run_dir, exp_name=exp_name),
                        log=log, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"federated training failed (exit {result.returncode}); see {log}"
        )
    experiment = run_dir / exp_name
    summary_path = experiment / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"training wrote no {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = summary.get("evaluation") or {}
    if "error" in evaluation:
        raise RuntimeError(f"post-training evaluation failed: {evaluation['error']}")
    final_adapter = Path(summary.get("final_adapter") or (experiment / "final-adapter"))
    return {
        "rounds": [row for row in summary.get("rounds") or [] if isinstance(row, dict)],
        "evaluation": {key: evaluation.get(key) for key in
                       ("before_passed", "after_passed", "cases")},
        "final_adapter": final_adapter,
    }


@dataclass
class Promotion:
    promoted: bool
    adapter: str | None
    reason: str
    best_after_passed: int | None


def promotion(*, after_passed: int | None, final_adapter: Path, state: State) -> Promotion:
    """Strictly better only: a tie keeps the adapter that is already known good."""
    if after_passed is None:
        return Promotion(False, state.best_adapter,
                         "the evaluation reported no after_passed count; nothing promoted",
                         state.best_after_passed)
    if state.best_after_passed is None:
        return Promotion(True, str(final_adapter),
                         f"first measured adapter: {after_passed} holdout case(s) passed",
                         after_passed)
    if after_passed > state.best_after_passed:
        return Promotion(True, str(final_adapter),
                         f"after {after_passed} passed, better than the best "
                         f"{state.best_after_passed}", after_passed)
    return Promotion(False, state.best_adapter,
                     f"after {after_passed} passed, not better than the best "
                     f"{state.best_after_passed}; kept {state.best_adapter}",
                     state.best_after_passed)


# --------------------------------------------------------------------------- #
# Real-pipeline evaluation
# --------------------------------------------------------------------------- #
def harness_databases(available, requested=None) -> list[str]:
    """Only the snapshots whose expected numbers the shared case file matches."""
    if requested:
        return list(requested)
    return [database for database in DEFAULT_EVAL_DATABASES if database in available]


def eval_command(app_python: str, database: str) -> list[str]:
    return [app_python, "-m", "softone_rag", "eval", "--real", "--database", database]


def default_report_path() -> Path:
    from softone_rag import real_eval

    return _ROOT / real_eval.REPORT_JSON


def summarize_real_report(report: dict) -> dict:
    """Aggregates only: per-case rows (SQL, answers) never leave this function."""
    if not isinstance(report, dict) or "pass_rate" not in report:
        raise ValueError("the real-eval report has no pass_rate")
    return {
        "pass_rate": report.get("pass_rate") or {},
        "per_kind": report.get("per_kind") or {},
        "failure_causes": report.get("failure_causes") or {},
        "latency_p95_ms": report.get("latency_p95_ms"),
    }


def run_harness(runner: SubprocessRunner, *, app_python: str, databases, logs: Path,
                report_path: Path) -> dict:
    """One real evaluation per snapshot; the CLI writes its own report each time."""
    results: dict[str, dict] = {}
    for database in databases:
        log = logs / f"eval-{database}.log"
        Path(report_path).unlink(missing_ok=True)
        result = runner.run(eval_command(app_python, database), log=log, cwd=_ROOT)
        entry: dict = {"exit_code": result.returncode}
        try:
            entry.update(summarize_real_report(
                json.loads(Path(report_path).read_text(encoding="utf-8"))
            ))
        except (OSError, ValueError) as error:
            entry["error"] = f"no usable eval report: {error}"
        else:
            if result.returncode not in (0, 1):
                entry["error"] = f"eval exited {result.returncode}; see {log}"
        results[database] = entry
    return {"databases": list(databases), "results": results, "skipped": None}


# --------------------------------------------------------------------------- #
# The one report
# --------------------------------------------------------------------------- #
def _score(passed, cases) -> str:
    if passed is None:
        return "n/a"
    return f"{passed}/{cases}" if cases is not None else f"{passed}"


def _loss(side: dict) -> str:
    first, last = side.get("mean_loss_first"), side.get("mean_loss_last")
    if first is None or last is None:
        return "n/a"
    return f"{float(first):.4f} → {float(last):.4f}"


def rounds_table(rounds: list[dict]) -> str:
    lines = [
        "| Round | PC loss first → last | Mac loss first → last | Seconds |",
        "| --- | --- | --- | --- |",
    ]
    for row in rounds:
        lines.append(
            f"| {row.get('round')} | {_loss(row.get('pc') or {})} "
            f"| {_loss(row.get('mac') or {})} | {row.get('seconds')} |"
        )
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    counts = report["export"]["counts"]
    training = report["training"]
    promotion = report["promotion"]
    harness = report["harness"]
    services_info = report["services"]
    skipped = (counts["skipped_guard"] + counts["skipped_duplicate"]
               + counts["skipped_holdout"])

    lines = [f"# Nightly {report['date']}", ""]
    lines.append(f"Examples exported: {counts['written']} "
                 f"(seed {counts['seed']}, confirmed {counts['confirmed']}, "
                 f"skipped {skipped})")
    lines.append("")
    lines.append(f"Training: {'yes' if training['trained'] else 'no'} — {training['reason']}")
    if training["error"]:
        lines.append("")
        lines.append(f"Training error: {training['error']}")
    if training["trained"]:
        lines.append("")
        lines.append("Holdout: before "
                     f"{_score(training['before_passed'], training['cases'])} → after "
                     f"{_score(training['after_passed'], training['cases'])}")
        lines.append(f"Promoted: {'yes' if promotion['promoted'] else 'no'} — "
                     f"{promotion['reason']}")
        lines.append("")
        lines.append("## Rounds, both machines merged")
        lines.append("")
        lines.append(rounds_table(training["rounds"]) if training["rounds"]
                     else "No round finished.")
    lines.append("")
    lines.append(f"Duration: {report['duration_seconds'] / 60:.1f} min")

    lines.append("")
    lines.append("## Real evaluation")
    lines.append("")
    if harness.get("skipped"):
        lines.append(harness["skipped"])
    else:
        evaluated = harness.get("databases") or []
        lines.append(f"Snapshots: {', '.join(evaluated)} ({EVAL_SCOPE_NOTE}).")
        lines.append("")
        lines.append("| Snapshot | Kind | Passed | Total | Pass rate |")
        lines.append("| --- | --- | --- | --- | --- |")
        for database, entry in harness["results"].items():
            kinds = entry.get("per_kind") or {}
            if not kinds:
                lines.append(f"| {database} | all | n/a | n/a | "
                             f"{entry.get('error', 'no results')} |")
                continue
            for kind, counts_kind in kinds.items():
                lines.append(
                    f"| {database} | {kind} | {counts_kind.get('passed')} "
                    f"| {counts_kind.get('total')} | {counts_kind.get('ratio')} |"
                )
        causes: dict[str, int] = {}
        for entry in harness["results"].values():
            for cause, count in (entry.get("failure_causes") or {}).items():
                causes[cause] = causes.get(cause, 0) + int(count)
        if causes:
            lines.append("")
            lines.append("Failure causes: "
                         + ", ".join(f"{cause} ×{count}"
                                     for cause, count in sorted(causes.items())))
        errors = [f"{database}: {entry['error']}"
                  for database, entry in harness["results"].items() if entry.get("error")]
        for error in errors:
            lines.append(f"- {error}")

    lines.append("")
    lines.append("## Services")
    stopped = services_info.get("stopped") or []
    if not stopped:
        lines.append("Nothing was stopped; both machines were left alone.")
    else:
        lines.append("Stopped: " + ", ".join(stopped))
        failed = services_info.get("failed_to_start") or []
        lines.append(f"Did not come back: {', '.join(failed)}" if failed
                     else "All came back.")
        health = services_info.get("llama_health") or {}
        if health.get("checked"):
            lines.append("llama-split /health: "
                         + ("answer received" if health.get("healthy")
                            else health.get("error", "no answer")))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="One nightly federated training and evaluation cycle on PC + Mac.",
    )
    parser.add_argument("--home", default=str(DEFAULT_HOME),
                        help="state, lock and experiments directory")
    parser.add_argument("--date", default=time.strftime(DATE_FORMAT),
                        help="YYYYMMDD label for this run")
    parser.add_argument("--min-new", type=int, default=DEFAULT_MIN_NEW,
                        help="new confirmed examples needed before training")
    parser.add_argument("--force-train", action="store_true",
                        help="train even when the confirmed count did not grow")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; change nothing on either machine")
    parser.add_argument("--app-python", default=DEFAULT_APP_PYTHON,
                        help="python of the app environment")
    parser.add_argument("--mac-host", default=DEFAULT_MAC_HOST, help="ssh host of the Mac")
    parser.add_argument("--databases", nargs="+", default=None,
                        help="snapshots for the real evaluation "
                             f"(default: {' '.join(DEFAULT_EVAL_DATABASES)})")
    return parser.parse_args(argv)


def plan_lines(*, args, run_dir: Path, exp_name: str, state: State) -> list[str]:
    from softone_rag import snapshots

    export_path = run_dir / "train.jsonl"
    lines = [
        f"Nightly plan for {args.date} (dry run)",
        f"  state: {state.last_trained_confirmed} confirmed example(s) at the last "
        f"trained run, best adapter {state.best_adapter or 'none'} "
        f"(after_passed {state.best_after_passed})",
        f"  export: {' '.join(export_command(args.app_python, export_path))}",
        f"  would train only with at least {args.min_new} new confirmed example(s)"
        + (" (forced with --force-train)" if args.force_train else ""),
        "  would stop, in this order:",
    ]
    ordered = services(args.mac_host)
    lines += [f"    {' '.join(service.stop)}" for service in ordered]
    lines.append("  would restore, in this order:")
    lines += [f"    {' '.join(service.start)}" for service in reversed(ordered)]
    lines.append("  would train: "
                 + " ".join(coordinator_command(app_python=args.app_python,
                                                data=export_path, run_dir=run_dir,
                                                exp_name=exp_name)))
    lines.append(f"  would wait for {LLAMA_HEALTH_URL} (up to "
                 f"{HEALTH_TIMEOUT_S / 60:.0f} min)")
    for database in harness_databases(list(snapshots.DATABASES), args.databases):
        lines.append(f"  would evaluate: {' '.join(eval_command(args.app_python, database))}")
    lines.append(f"  would write {run_dir / 'report.md'} and "
                 f"{run_dir / 'report.json'}")
    lines.append("Dry run: nothing was stopped, exported, trained or evaluated.")
    return lines


def main(argv: list[str] | None = None, *, runner=None, probe=None,
         report_path: Path | None = None) -> int:
    """One cycle. ``runner``/``probe``/``report_path`` exist so tests stay offline."""
    args = parse_args(argv)
    from softone_rag import snapshots

    home = Path(args.home).expanduser()
    run_dir = home / "experiments" / f"{RUN_PREFIX}-{args.date}"
    exp_name = f"{RUN_PREFIX}-{args.date}"
    export_path = run_dir / "train.jsonl"
    logs = run_dir / "logs"
    state_path = home / "nightly-state.json"
    lock_path = home / "nightly.lock"
    runner = runner if runner is not None else SubprocessRunner()
    report_path = Path(report_path) if report_path else default_report_path()
    started = time.monotonic()

    try:
        state = State.load(state_path)
    except RuntimeError as error:
        print(f"nightly aborted: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n".join(plan_lines(args=args, run_dir=run_dir, exp_name=exp_name,
                                   state=state)))
        return 0

    lock = acquire_lock(lock_path)
    if lock is None:
        print(f"Nightly already running (lock {lock_path}); nothing to do.")
        return 0
    with lock:
        run_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(run_dir, 0o700)
        logs.mkdir(parents=True, mode=0o700, exist_ok=True)

        try:
            counts = export_examples(runner, app_python=args.app_python, out=export_path,
                                     log=logs / "export.log")
        except RuntimeError as error:
            print(f"nightly aborted: {error}", file=sys.stderr)
            return 1

        should_train, reason = decide(confirmed=counts["confirmed"], state=state,
                                      min_new=args.min_new, force=args.force_train)
        training = {"trained": False, "reason": reason, "error": None, "rounds": [],
                    "before_passed": None, "after_passed": None, "cases": None,
                    "final_adapter": None}
        promotion_info = {"promoted": False, "reason": "no training ran",
                          "adapter": state.best_adapter}
        stopped: list[str] = []
        failed_restores: list[str] = []
        health: dict = {"checked": False}

        if should_train:
            ordered = services(args.mac_host)
            try:
                stopped = stop_services(runner, ordered, logs)
                outcome = train(runner, app_python=args.app_python, data=export_path,
                                run_dir=run_dir, exp_name=exp_name,
                                log=logs / "coordinator.log")
                evaluation = outcome["evaluation"]
                training.update(trained=True, rounds=outcome["rounds"],
                                before_passed=evaluation["before_passed"],
                                after_passed=evaluation["after_passed"],
                                cases=evaluation["cases"],
                                final_adapter=str(outcome["final_adapter"]))
                decision = promotion(after_passed=evaluation["after_passed"],
                                     final_adapter=outcome["final_adapter"], state=state)
                promotion_info = {"promoted": decision.promoted,
                                  "reason": decision.reason,
                                  "adapter": decision.adapter}
                if decision.promoted:
                    state.best_adapter = decision.adapter
                    state.best_after_passed = decision.best_after_passed
                state.last_trained_confirmed = counts["confirmed"]
            except (OSError, RuntimeError, ValueError) as error:
                training["error"] = str(error)
                print(f"nightly: training failed: {error}", file=sys.stderr)
            finally:
                # Always, in reverse order: Mac RPC, llama-split, both chats.
                failed_restores = restore_services(runner, ordered, logs)
            health = wait_for_llama_health(probe=probe or llama_health)

        if health.get("checked") and not health.get("healthy"):
            harness = {"databases": [], "results": {}, "skipped": "llama-split did not "
                       "answer /health, so the real evaluation was skipped"}
        else:
            harness = run_harness(
                runner, app_python=args.app_python,
                databases=harness_databases(list(snapshots.DATABASES), args.databases),
                logs=logs, report_path=report_path,
            )

        state.last_run = args.date
        try:
            state.save(state_path)
        except OSError as error:
            print(f"nightly: could not save {state_path}: {error}", file=sys.stderr)

        report = {
            "date": args.date,
            "run_dir": str(run_dir),
            "export": {"path": str(export_path), "counts": counts},
            "decision": {"train": should_train, "reason": reason},
            "training": training,
            "promotion": promotion_info,
            "harness": harness,
            "services": {"stopped": stopped, "failed_to_start": failed_restores,
                         "llama_health": health},
            "duration_seconds": round(time.monotonic() - started, 1),
        }
        write_private(run_dir / "report.json", json.dumps(report, indent=2) + "\n")
        write_private(run_dir / "report.md", render_markdown(report))
        print(f"Nightly report: {run_dir / 'report.md'}")
        return 1 if (training["error"] or failed_restores) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"nightly aborted: {error}", file=sys.stderr)
        sys.exit(1)
