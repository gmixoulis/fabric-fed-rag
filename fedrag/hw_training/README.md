# Training notes

Two guarded QLoRA experiments live here. Neither is a deployment step: an adapter
is only useful after `uv run python -m softone_rag eval` says it is.

| Path | What it is |
| --- | --- |
| `train_softone_qwen.py` | the original single-machine run (this PC only) |
| `softone_training.py` | shared guards, chunked loss, prompt/example building |
| `fed/` | federated (FedAvg) LoRA training across this PC and the MacBook |

Everything here needs a training virtualenv with torch, transformers and peft; the
repo's own `uv` environment stays free of them, which is why these files are not
part of the installed `softone_rag` package.

## Federated training

Federated training trains the same LoRA adapter on two machines at once and
averages the updates each round (FedAvg), so neither machine ever sees the other
machine's examples. The target task is unchanged: natural-language question to
one read-only SQLite `SELECT` over the SoftOne-shaped schema.

### What runs where

- **PC (GTX 1660 SUPER, 6 GiB, CUDA)** — `fed/coordinator.py` and one client per
  round. The coordinator renders every (prompt, target) pair once with
  `softone_rag.graph.build_prompt` and the real catalog, splits it into two
  shards, drives the rounds, and averages the adapters. It is plain python in the
  repo's `uv` environment: no torch, no model, no CUDA context. The PC client
  trains QLoRA on the 4-bit bitsandbytes checkpoint at
  `~/.local/share/softone-chat/qwen35-4b-bnb` (`Qwen3_5ForCausalLM`, text only).
- **MacBook Air M4 (16 GiB, MPS)** — `fed/client.py` only, from
  `~/opt/fedtrain/.venv` (python 3.12, torch 2.14.1, transformers 5.18.0,
  peft 0.21.3). It trains in bf16 from `~/opt/fedtrain/models/qwen35-4b`, the
  original multimodal `Qwen/Qwen3.5-4B` checkpoint. There is no `softone_rag` on
  the Mac and none is needed: a shard is plain prompt text plus raw SQL, and each
  client applies its own base model's chat template and tokenizer.
- **Over ssh** — the coordinator is the only thing that talks to the Mac. It
  copies the shard, `client.py`, `fedavg.py`, `softone_training.py`,
  `mac_run.sh`, the round config and the current global adapter into
  `~/opt/fedtrain/runs/<experiment>/round-<k>/`, starts the client there, and
  copies the resulting adapter back.

**The data shards are copied to the Mac.** `runs/<experiment>/round-<k>/shard.jsonl`
holds that round's prompt and target SQL text on the Mac's disk, and it stays
there after the round. It is the 16 seed questions from
`eval/softone_train_seed.jsonl` and nothing from the business snapshots. Delete
`~/opt/fedtrain/runs` when an experiment is finished.

### Memory guards

Both clients refuse to start rather than run unbounded:

- **PC**: every training process must be started by `~/.local/bin/train-safe`,
  which puts it in a systemd user scope with `MemoryMax`/`MemorySwapMax` and a
  single-job lock. The client re-checks that it is inside a finite
  `agent-training.scope` (`AGENT_TRAINING_GUARDED=1`, `memory.max` and
  `memory.swap.max` both set) and refuses on CUDA otherwise, then refuses again
  unless at least 4 GiB of GPU memory is free. Never launch it with bare
  `python`; the coordinator always goes through `train-safe`.
- **Mac**: `fed/mac_run.sh` sets `torch.mps.set_per_process_memory_fraction`
  (`mps_memory_fraction`, default 0.75 of the recommended working set) before the
  model loads and refuses to run if that call is unavailable. A watchdog then
  polls the client's resident size every 2 s and kills it above
  `mac_rss_limit_gb` (default 13) or after `round_timeout_s`, exiting non-zero
  with the reason. A killed client aborts the whole run; nothing is retried
  automatically, because a silent retry would hide the memory regression.
- On the Mac the model is loaded straight onto MPS from a text-only copy of the
  checkpoint (see below); a CPU copy plus an MPS copy would double the resident
  size and trip the watchdog.

### The checkpoint layouts, and why the adapters still match

The Mac repo is `Qwen3_5ForConditionalGeneration`: its text tower is
`model.language_model.layers.N...`. The PC checkpoint is already text-only:
`model.layers.N...`. On the Mac the client therefore extracts the language-model
tensors once into `~/opt/fedtrain/models/qwen35-4b-text` (a normal text-only
checkpoint, reused by later rounds) and trains only that. Adapters are always
saved with the text-model key layout: `fed/fedavg.py` normalises the
`language_model` segment on save and on load, and the average refuses mismatched
key sets or shapes instead of quietly dropping tensors.

Both clients seed the LoRA `A` basis deterministically from the config seed and
the tensor name, with `B` left at zero, so the two round-1 adapters share a
basis and the first average means something. From round 2 onward both clients
start from the same averaged adapter.

The Mac checkpoint must be complete before the first round. The client checks the
safetensors index and fails with the missing shard names (and the download
command) rather than loading a partial download.

### Running it

```bash
# PC, from the repo root, in the repo environment (no torch here).
uv run python training/fed/coordinator.py --dry-run   # 1 round, 1 step, 2 examples per shard
uv run python training/fed/coordinator.py             # config.json: 3 rounds

# The whole nightly cycle: export, decide, train on both machines, restore the
# services, real-eval, one merged report. See "The nightly cycle" below.
uv run python training/fed/nightly.py --dry-run       # print the plan only
```

The clients themselves run in the training venv (`--pc-python`) and in
`~/opt/fedtrain/.venv` on the Mac (`--mac-python`); the coordinator only builds
their command lines.

The experiment lands in `~/.local/share/softone-chat/experiments/fed-<timestamp>`
(mode 0700, never overwritten):

```
config.json          the effective config for this run
shard-pc.jsonl       the PC shard
shard-mac.jsonl      the shard that was copied to the Mac
round-<k>/pc.log     client stdout/stderr
round-<k>/mac.log    ssh stdout/stderr
round-<k>/round.json losses, examples, seconds and device for both clients
round-<k>/global/    the averaged adapter after round <k>
final-adapter/       a normal PEFT adapter directory for the PC base model
summary.json         every round, the final adapter path and, with --evaluate, the counts
eval.json            with --evaluate: before/after per-case records and both counts
failure.json         written instead of the above if a client failed
```

Useful flags: `--rounds N`, `--exp-name NAME`, `--mac-host ALIAS`,
`--mac-root ~/opt/fedtrain`, `--mac-python ~/opt/fedtrain/.venv/bin/python`,
`--pc-python .../train-venv/bin/python`, `--train-safe ~/.local/bin/train-safe`.
`--dry-run` really does run both clients, it just makes the round tiny; it is the
smoke test before a longer run. Anything else needs `tests/`-level checks only.

A round is: push, start both clients at the same time, wait, require both to exit
0, fetch the Mac adapter, and average weighted by `num_examples`. A non-zero exit,
a watchdog kill or a timeout stops the run and leaves `failure.json`; the
coordinator does not retry.

### Configuration

`fed/config.json` is copied to both clients for every round, so both sides always
agree:

| Key | Meaning |
| --- | --- |
| `rounds`, `local_steps`, `lr`, `seed` | the training loop |
| `lora.r/alpha/dropout/target_modules` | must match exactly; checked on every average |
| `grad_accumulation` | micro-batches per optimiser step (4, as in the single-machine run) |
| `max_length` | hard cap on tokens per example; longer examples abort the round |
| `shard_weights` | `[pc, mac]` share of the examples, default even |
| `mps_memory_fraction` | Mac allocator cap, 0.75 by default |
| `mps_dtype` | `bfloat16` by default, `float16` if MPS bf16 misbehaves |
| `mac_rss_limit_gb` | watchdog limit on the Mac |
| `round_timeout_s` | both the watchdog timeout and the coordinator's wait |
| `base_pc`, `base_mac` | the two base checkpoints |

### Optional DoRA

Add `"use_dora": true` to the `lora` section of `config.json` to train with PEFT's weight-decomposed LoRA (`LoraConfig(use_dora=True)`); the key is optional, must be a bool, and defaults to `false`. Both clients use the same value and the coordinator rejects an adapter whose `use_dora` differs, so DoRA magnitude vectors are saved, checked and averaged with the other LoRA tensors. Do not assume a PEFT DoRA adapter converts with llama.cpp's `convert_lora_to_gguf`: merge it into the base model before GGUF export unless that conversion has been verified for your PEFT/llama.cpp versions.

### Held-out evaluation

An adapter is only useful if it answers questions it was not trained on. The
held-out set is `eval/softone_holdout.jsonl`: twelve `{id, question, sql}` cases
over the real snapshot tables, half of them plain counts and dates, half using
the taught codes (`TRDR.SODTYPE`, `FINDOC.SOSOURCE`, `FINDOC.ISCANCEL`,
`MTRL.SODTYPE`), three of them in Greek. The codes those cases compare against
are listed in `eval/softone_verified_codes.json`, so the guard can accept a
correct code and still reject an invented one.

Training data comes from `softone-rag export-training OUT.jsonl` (the seed plus
the answers a human confirmed, guard-checked, deduplicated, and with every
held-out question removed). Point a run at it with `--data OUT.jsonl`.

Score one adapter before and after attaching it:

```bash
~/.local/bin/train-safe ~/.local/share/softone-chat/train-venv/bin/python \
  training/fed/evaluate_adapter.py \
  --adapter <experiment>/final-adapter --out <experiment>/eval.json
```

It refuses to start outside `agent-training.scope` or with less than 4 GiB of
free GPU memory, loads the base the way `fed/client.py` does, scores the base,
attaches the adapter with PEFT and scores again. The report is written mode 0600
and never overwritten, and it fails loudly when a held-out question also appears
in `--train-data` (default `eval/softone_train_seed.jsonl`).

The coordinator can do that for you at the end of a run with `--evaluate` (off
by default): it starts the same command under `train-safe`, and copies
`before_passed`, `after_passed` and `cases` into `summary.json`.

### The nightly cycle

`fed/nightly.py` is the one scheduled entry point, and it produces exactly one
report for both machines. It runs in the app environment (no torch):

1. Takes `~/.local/share/softone-chat/nightly.lock` and exits 0 if another run
   holds it.
2. Exports to `experiments/nightly-YYYYMMDD/train.jsonl` (0600, never overwritten).
3. Trains only when the export carries at least `--min-new` (20) more confirmed
   examples than `nightly-state.json` recorded for the last trained run, or with
   `--force-train`.
4. Stops both chat services, `llama-split` and the Mac's `local.llama.rpc` so the
   GPUs are free on both sides, then runs the coordinator with `--evaluate`.
   Promotion is strict: `after_passed` must beat `best_after_passed`, otherwise
   the previously best adapter stays recorded. No adapter is ever deleted.
5. Restores everything in a `finally` (Mac RPC, `llama-split`, then the chats) and
   waits up to 15 min for `http://127.0.0.1:8095/health`.
6. Runs `eval --real` per snapshot and writes `report.md` + `report.json` (0600):
   counts, whether it trained and why, one merged loss table for both machines,
   holdout before/after, promotion, real-eval pass rate per kind, failure causes,
   duration and any service that did not come back. Aggregate numbers only, never
   a result row, an answer or a memory text.

`--dry-run` prints the whole plan (services and commands) and changes nothing.
`--databases` widens the real evaluation, but the expected numbers in
`eval/real_cases.jsonl` only match `softone_reputation`, so that is the default.
The unit and timer live in `deploy/systemd/` (`OnCalendar=*-*-* 03:00`,
`Persistent=false`); they are not installed by the repo.

### Tests

`uv run pytest -q` covers the offline parts: FedAvg weighting on tiny numpy
tensors, key normalisation in both directions with a loud failure on mismatch,
the deterministic shard split, the coordinator's `train-safe`/`ssh`/`scp`
commands with subprocess mocked, the client's refusal to run on CUDA outside
a finite cgroup or on MPS without a memory fraction, the export rules (confirmed
only, guard rejections skipped, dedupe, held-out exclusion, 0600, no overwrite)
and the shared case scoring against a tiny temporary SQLite database. The nightly
cycle is covered too: the lock, the threshold, stop/restore order (also when the
coordinator fails), strict promotion, dry-run, and a report that keeps aggregates
only and is 0600. No test loads a model, opens a port, runs ssh or reads `data/`.
