## Current Session State

### Completed

- [x] Research Hyperledger Fabric 2.x + Federated LoRA + Decentralized Sharded RAG feasibility and 2024–2026 academic literature
- [x] Audit and `git pull` latest `softone-rag` (`4879aa4`)
- [x] Initialize standalone repository `/Users/gmixoulis/Desktop/my-projects/fabric-fed-rag` and push online to `https://github.com/gmixoulis/fabric-fed-rag`
- [x] Build Python + TypeScript Raft + mTLS + minimal validation ledger (`fedrag/raft_ledger.py`, `chaincode/validator.ts`, `fedrag/certgen.py`)
- [x] Port and extend Federated LoRA training (`fedavg.py`, `trainer.py` with `fedavg` & `ffa` modes, `hw_training/`) and the PDF/RAG store (`shard_store.py`, `embed.py`, `node_service.py`)
- [x] Include agent instruction files (`AGENTS.md`, `.trail/learnings/`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, `docs/RESEARCH-LOG.md`, `docs/RESEARCH_PLAN.md`)
- [x] Add modular CLI runner (`./run.sh`) and comprehensive test suite
- [x] WP1: three organisations on the ledger — `Peer3Org`/`node3-peer` certs and connection profile, configured Raft membership with 2-of-3 commit and election, configurable `LORA_ROUND_FINALIZE` majority quorum, offline three-node mTLS tests (`docs/EXPERIMENTS.md` E8)
- [x] WP2: one Ollama per node — `ollama-node1..3` (pinned image, `mem_limit: 2g`, shared read-only model volume, internal network), `FEDRAG_OLLAMA_URL` per node, `embedding_model_digest` on every `DOC_MANIFEST`, digest agreement checked against peers and the ledger before an ingest writes
- [x] WP3: thirds instead of halves — `ShardRing` hash assignment over the member list (N = 3; N = 2 unchanged), metadata stubs for every other shard, ONE ring backup replica (1→2, 2→3, 3→1) with `read_backup`/`ingest_restored_backup`/`POST /shard/restore`, `FEDRAG_MEMBERS` member list replacing `peer_url`, scatter-gather with per-peer timeout that reports `missing_nodes`/`missing_shards`, federated rounds over all reachable members with participants recorded, `aggregation=average|summed` (server lr = participants), `ipfs-node3`/`node3-peer`/`node3-data` in compose, `./run.sh status` over three shards (`docs/EXPERIMENTS.md` E9)

### In Progress

- Planned 2026-11-20: three nodes, one Ollama per node, data in thirds, Mac as third node later. See `docs/PLAN-3-NODES-OLLAMA.md`. WP1–WP3 done; WP4 (three-node federated runs on the GPU) next, then WP5 (node 3 on the Mac) and WP6 (write-up).

### Not verified yet (WP3)

- No Docker container was started and no hardware training was run: the host had under 4 GB
  of free memory. `docker compose config -q` validates the three-node topology; the
  three-node cluster behaviour is verified offline over ephemeral mTLS sockets only.
- Recall@k with 3, 2 and 1 nodes alive against the synthetic ERP replica (WP4 numbers) is
  not measured; the offline test asserts equality with a single index and the reported
  degradation with one node down.

### Still two-node on purpose

- `tests/test_raft_mtls_cluster.py` (the N = 2 regression) and the recorded two-node numbers
  in `docs/REAL_EVAL_RESULTS.md`.
- The mode key `decentralized_50_50_rag` in `scripts/run_real_eval.py` and its Excel exporter,
  even though the corpus is now routed to three hash shards.
- `fedrag/hw_training/` trainers still describe one PC shard and one Mac shard (WP5 adds the
  third host).
- E1–E8 in `docs/EXPERIMENTS.md` and `docs/RESEARCH_PLAN.md` are historical two-node results
  and are not rewritten.

### Decisions Made

- Standalone repo at `/Users/gmixoulis/Desktop/my-projects/fabric-fed-rag` (`https://github.com/gmixoulis/fabric-fed-rag`) using Python + TypeScript (`chaincode/validator.ts` + `fedrag/`).
- Included both `fedavg` and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`) modes in `fedrag/trainer.py`.
- Included `AGENTS.md` and `.trail/learnings/` so future coding agents have full context and non-negotiable gates.
- Shard assignment is `sha256("fabric-fed-rag/shard-ring/v1:<doc_id>") mod N` over the sorted
  member list: reproducible from configuration alone, no coordination, and N = 2 reduces to
  the original 50/50 split. The salt is part of the hash so a future re-split cannot move
  documents that were already committed.
- Ring order defines the single backup replica (successor), so one lost node is recoverable
  and no node stores two replicas.
- `aggregation=summed` means `previous + N * (mean - previous)` with N = participants, the
  softone-rag E-2026-11-18 definition of summed updates; `average` stays the default and is
  bit-identical to the old weighted FedAvg.

### Blocked

- None
