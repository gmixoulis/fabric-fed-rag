## Current Session State

### Completed

- [x] Research Hyperledger Fabric 2.x + Federated LoRA + Decentralized Sharded RAG feasibility and 2024–2026 academic literature
- [x] Audit `softone-rag` components (`training/fed/`, `embed.py`, `memory.py`) for reuse
- [x] Initialize standalone repository `/Users/gmixoulis/Desktop/my-projects/fabric-fed-rag`
- [x] Build Python + TypeScript Raft + mTLS + minimal validation ledger (`fedrag/raft_ledger.py`, `chaincode/validator.ts`, `fedrag/certgen.py`)
- [x] Port and extend Federated LoRA training (`fedavg.py`, `trainer.py` with `fedavg` & `ffa` modes, `hw_training/`) and 50/50 Sharded PDF/RAG store (`shard_store.py`, `embed.py`, `node_service.py`)
- [x] Verify 2-Docker-node mTLS testbed (`node1-maestro` and `node2-peer`) and execute `scripts/run_demo.py` on MacBook Air M4 (all 22 Raft blocks, 50/50 shards, 3 Federated LoRA rounds, scatter-gather RAG, and tamper rejection verified)
- [x] Commit standalone repository as `gmixoulis <gmixoulis@gmail.com>`

### In Progress

- None

### Decisions Made

- Standalone repo at `/Users/gmixoulis/Desktop/my-projects/fabric-fed-rag` using Python + TypeScript (`chaincode/validator.ts` + `fedrag/`).
- Included both `fedavg` and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`) modes in `fedrag/trainer.py`.
- Included comprehensive research plan & 2024–2026 paper comparison in `docs/RESEARCH_PLAN.md`.

### Blocked

- None
