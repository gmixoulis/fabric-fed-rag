# Plan: three nodes, each with its own Ollama (one PC now, the Mac later)

Written 2026-11-20 for a separate agent session. Read `AGENTS.md`, `docs/ARCHITECTURE.md` and
`state.md` first. This repository is PUBLIC: no client data, no real snapshots, no result rows.
Use only the synthetic replica and the public question sets (see "Data").

## Goal

Today the system has two nodes that share the host's Ollama. The target is a three-organisation
consortium in which every node owns its model runtime, its third of the data and its vote on
the ledger:

1. Now: three node containers on the PC, each with its own Ollama container.
2. Later: the third node moves to the MacBook (same certificates and data third), with no change
   to the ledger rules.

Research question: with the data split in thirds and every step recorded on the ledger, how do
retrieval quality, federated LoRA accuracy, wall time and ledger overhead compare with two
nodes and with one machine? The one-machine and two-machine numbers already exist
(`softone-rag/docs/EXPERIMENTS.md`: 83/94 one machine, 81-85/94 two machines with summed updates).

## Hardware limits that shape the design (measured, not assumed)

- PC: 15 GB RAM with about 5 GB free when the SoftOne chat is up, swap usually full, one 6 GB GPU.
- So three Ollama containers cannot each hold a chat model. Each node's Ollama holds only the
  embedding model (bge-m3, about 1.2 GB, CPU). That is 3.6 GB; check `free -g` before starting
  and stop if less than 5 GB is available. Give every Ollama container `mem_limit: 2g`.
- Text generation for RAG answers: one small model (gemma4:e4b, 330 MB on the GPU) served by
  node 1's Ollama only, and only when the GPU is free. Nodes 2 and 3 answer retrieval; node 1
  synthesises. Record this asymmetry in the paper as a hardware limit.
- Training: every training process runs through `~/.local/bin/train-safe` on the host, one job at
  a time. A container cannot bypass it: the node service asks a host-side runner for its round.
  Three nodes on one GPU therefore train one after the other; report wall time and the summed
  per-node time separately, and say that rounds were sequential.
- Never raise `train-safe` limits, never stop the SoftOne chat or other services to make room.
  If the guard refuses, wait or reduce (smaller batch, shorter max length) and write it down.
- The Mac is busy with a long training run until about 2026-11-22. Do not use it before that.

## Data

- Documents for sharded RAG: public or synthetic PDFs only (the existing test documents in this
  repo). Split by a seeded hash of `doc_id` into three shards.
- Training and evaluation: `softone-rag/paper/release/erp-replica-release.zip` (synthetic ERP
  replica generator, 94 held-out text-to-SQL cases, 47 multi-step and 32 company questions keyed
  to the replica). The owner has not published that bundle yet, so unpack it into a gitignored
  directory (`data/replica-release/`) and read it from there; do not commit its contents here
  until the owner publishes it. Never copy from `softone-rag/eval` or any snapshot.
- Training seed: 124 synthetic question/SQL pairs, split in three by stratified groups (plain,
  taught-code, join, refusal) so that joins do not all land on one node.

## Work packages (each ends with tests passing and a short entry in `docs/EXPERIMENTS.md`)

### WP1. Three organisations on the ledger
- `fedrag/certgen.py`, `connection-profile.json`: add `Peer3Org` with its own certificate.
- Raft with three members: leader election and commit need 2 of 3. Add a test that stops one
  node and checks that the other two still commit, and that a stopped node catches up.
- Chaincode (`chaincode/validator.ts` and the Python twin): `LORA_ROUND_FINALIZE` requires
  submissions from a quorum that is a parameter (2 of 3), not the constant 2.
- Gate: existing 7 tests plus the new ones.

### WP2. One Ollama per node
- `docker-compose.yml`: services `ollama-node1..3` (image pinned to the host's Ollama version,
  `mem_limit: 2g`, no published ports, on the internal network only), one shared read-only
  volume for model files so the 1.2 GB model is stored once, and `FEDRAG_OLLAMA_URL` per node
  pointing at its own Ollama instead of `host.docker.internal`.
- Pull the embedding model once into the shared volume; verify each node embeds through its own
  container (log the model digest on the ledger with `DOC_MANIFEST` so mixed versions are caught).
- GPU: only `ollama-node1` gets the GPU, and only for the small chat model.
- Gate: `./run.sh up` on a machine with 5 GB free, `docker stats` under the limits, embedding
  latency per node recorded.

### WP3. Thirds instead of halves
- `fedrag/shard_store.py`: generalise 50/50 to N shards (N = 3): each node pins and indexes its
  third and keeps metadata stubs plus one backup replica of another third (ring: 1 backs up 2,
  2 backs up 3, 3 backs up 1), so one node can be lost.
- IPFS: add `ipfs-node3` to the private swarm.
- Scatter-gather query over three nodes with a timeout per node; a missing node degrades recall
  but does not fail the query. Report recall@k with 3, 2 and 1 nodes alive.
- Gate: retrieval recall on the test documents equals the two-node baseline when all are alive.

### WP4. Federated LoRA over three nodes
- Reuse `softone-rag/training/fed` ideas already ported here: summed updates (server learning
  rate equal to the number of nodes with disjoint data; measured better than averaging),
  rank 16 on q/k/v/o, 4 rounds.
- Each node's adapter hash goes on the ledger before aggregation; the aggregate hash after.
  A node that submits a hash that does not match its file is rejected (extend the tamper test).
- Runs to make, each scored on the 94 held-out cases against the replica, three seeds if time
  allows (single-run differences of 3-4 cases are noise, see softone-rag E-2026-11-20):
  1. one node, all 124 examples (baseline, already measured: 83);
  2. three nodes, thirds, summed updates, sequential on the PC GPU;
  3. three nodes with one node dropped in round 2 (quorum 2 of 3);
  4. FFA-LoRA instead of FedAvg, same split.
- Record wall time, per-node time, ledger time per round, bytes on the ledger.

### WP5. Move node 3 to the Mac (after 2026-11-22, when its training run is done)
- Same node image is not possible on macOS GPU: run the node service natively on the Mac
  (Python, MPS trainer with the RSS watchdog already in `fedrag/hw_training/`) and its own
  Ollama (already installed there). Use the Tailscale address; mTLS as for the other nodes.
- Then training is truly parallel for one of the three nodes: compare wall time with WP4.
- The Mac has 16 GB and cannot train while it serves the 26B chat model for SoftOne; agree a
  time window with the owner before each run.

### WP6. Write-up
- Table: 1 node / 2 nodes / 3 nodes (PC only) / 3 nodes (PC + Mac): accuracy, time, ledger
  overhead, recall with a node down. Threats: sequential training on one GPU, asymmetric
  generation, small data, single runs.
- Update `README.md`, `docs/ARCHITECTURE.md` (three-node diagram) and `state.md`.

## Order and size

WP1 and WP2 are independent (start both). WP3 needs WP1. WP4 needs WP1 and the GPU, so run it
at night or when the SoftOne chat is on the Mac. WP5 waits for the Mac. Rough effort: WP1 one
session, WP2 half a session, WP3 one session, WP4 two nights of GPU time, WP5 one session.

## Not in scope

Real Hyperledger Fabric peers (the repo models Fabric with a Raft + mTLS ledger; keep that and
say so), byzantine fault tolerance, differential privacy, more than three nodes, any client data.

## First commands for the new session

```bash
cd ~/Desktop/fabric-fed-rag && git pull && cat AGENTS.md state.md docs/PLAN-3-NODES-OLLAMA.md
free -g && nvidia-smi --query-gpu=memory.used --format=csv,noheader   # is there room?
./run.sh certs && ./run.sh build && uv run pytest -q                  # baseline before changes
```
