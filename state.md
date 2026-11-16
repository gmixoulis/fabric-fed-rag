## Current Session State

### Completed

- [x] Research Hyperledger Fabric 2.x + Federated LoRA + Decentralized Sharded RAG feasibility and 2024–2026 academic literature
- [x] Audit and `git pull` latest `softone-rag` (`4879aa4`)
- [x] Initialize standalone repository `/Users/gmixoulis/Desktop/my-projects/fabric-fed-rag` and push online to `https://github.com/gmixoulis/fabric-fed-rag`
- [x] Build Python + TypeScript Raft + mTLS + minimal validation ledger (`fedrag/raft_ledger.py`, `chaincode/validator.ts`, `fedrag/certgen.py`)
- [x] Port and extend Federated LoRA training (`fedavg.py`, `trainer.py` with `fedavg` & `ffa` modes, `hw_training/`) and 50/50 Sharded PDF/RAG store (`shard_store.py`, `embed.py`, `node_service.py`)
- [x] Include agent instruction files (`AGENTS.md`, `.trail/learnings/`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, `docs/RESEARCH-LOG.md`, `docs/RESEARCH_PLAN.md`)
- [x] Add modular CLI runner (`./run.sh`) and comprehensive test suite (`tests/test_pipeline.py`, `tests/test_fedavg_and_keys.py`, `tests/test_raft_mtls_cluster.py` — 7/7 passing)

### In Progress

- None

### Decisions Made

- Standalone repo at `/Users/gmixoulis/Desktop/my-projects/fabric-fed-rag` (`https://github.com/gmixoulis/fabric-fed-rag`) using Python + TypeScript (`chaincode/validator.ts` + `fedrag/`).
- Included both `fedavg` and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`) modes in `fedrag/trainer.py`.
- Included `AGENTS.md` and `.trail/learnings/` so future coding agents have full context and non-negotiable gates.

### Blocked

- None
