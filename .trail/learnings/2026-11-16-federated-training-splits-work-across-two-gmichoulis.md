---
title: Federated training splits work across two Python environments that do not overlap
author: Gmichoulis
date: 2026-11-16
cost: { minutes: 25, tokens: 64477 }
id: l-e0c50251
scope: repo
status: active
src: agent
touches:
  - training/fed/coordinator.py
  - training/fed/fedavg.py
---

## Decided
The federated coordinator runs in the app environment (it needs softone_rag/langgraph to build prompts), but anything touching torch or safetensors, including FedAvg itself, runs as a subprocess in the training venv. Clients apply their own base model's chat template because the PC (4-bit text-only) and Mac (bf16 multimodal) tokenizer files differ.

## Tried and ruled out
Averaging adapters inside the coordinator failed: the app environment has no torch/safetensors. Running the coordinator in the training venv also fails: it has no langgraph.

## Watch out
On the Mac, the RSS watchdog does not see Metal/GPU memory; the MPS memory fraction is the real GPU cap. On the PC, the process pgrep finds first is the train-safe launcher (no limits); verify the worker in agent-training.scope instead.

In the code:
- training/fed/coordinator.py federated_average -> `<train-venv python> training/fed/fedavg.py <pc> <mac> <out> <config> <base> <round>`
- training/fed/fedavg.py __main__ adds training/ to sys.path for softone_training
- PC worker scope: user@1000.service/app.slice/agent-training.scope (MemoryMax 6 GiB, swap 0.94 GiB)
