---
title: "The PC trainer never updated its adapter: fp16 gradients were NaN and the step was skipped silently"
author: Gmichoulis
date: 2026-11-17
branch: main
cost: { minutes: 38, tokens: 272047 }
id: l-a32ac26a
scope: repo
status: active
src: agent
touches:
  - fedrag/fedavg.py
  - fedrag/hw_training/fedavg.py
  - fedrag/hw_training/client.py
  - fedrag/raft_ledger.py
  - fedrag/node_service.py
---

## Decided
Start the CUDA GradScaler at scale 1 and raise when any gradient is non-finite. With that, a 32-step QLoRA run on the 8-example PC shard goes from loss 0.56 to 0.013; before, loss was flat and lora_B was exactly zero. DoRA is an opt-in flag (lora.use_dora), not the default: on the 12-case holdout QLoRA and QDoRA both scored 4/12 against 5/12 for the base model, DoRA took 2% longer and stores 48 tensors instead of 32.

## Tried and ruled out
- Blaming too few steps or a low learning rate for the flat loss: the weights had not moved at all (max|B| = 0), also in the earlier fed-test-16 round-1 PC adapter.
- Reading a 0/12 holdout score as model quality: every case had failed with ModuleNotFoundError for softone_rag, and evaluate_cases records that per case instead of stopping.

## Watch out
- Gradients are NaN at loss scale 65536 and 1024, finite at 16 and 1 (and in fp32). GradScaler halves its scale after each skipped step, so a round with few optimizer steps never recovers.
- The hardware scripts must not put the parent package directory on sys.path: fedrag/fedavg.py then shadows fedrag/hw_training/fedavg.py and the client crashes at save, after training.
- Run evaluate_adapter.py with the softone-rag src and app site-packages on PYTHONPATH.
- A training loss that falls says nothing about held-out accuracy here; check max|B| and the holdout.

In the code: fedrag/hw_training/client.py train_local (GradScaler init_scale=1.0, finite-gradient check), sys.path setup in client.py, coordinator.py, nightly.py, evaluate_adapter.py, fedavg.py.
