# FabricFedRAG (`fabric-fed-rag`)

Decentralized **50/50 Sharded PDF RAG** and **Federated LoRA Training (`FedAvg` & `FFA-LoRA`)** coordinated over a **Python / TypeScript Raft + mutual TLS (mTLS)** ledger (modeled on Hyperledger Fabric 2.x Raft Orderer + private data hash commitments).

See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the full research plan, hypotheses, and 2024–2026 literature review.

## Architecture

1. **Raft + mTLS Ledger & TypeScript/Python Chaincode (`fedrag/raft_ledger.py`, `chaincode/validator.ts`, `fedrag/certgen.py`)**
   - Every node authenticates with X.509 certificates issued by the consortium CA (`fedrag/certgen.py`) over TLS 1.3 (`ssl.CERT_REQUIRED`).
   - `node1-maestro` runs the Raft Leader / Orderer and Federated Coordinator.
   - `node2-peer` runs the Raft Follower and Peer RAG/Trainer.
   - Minimal chaincode state validation enforces:
     - `DOC_MANIFEST`: commits `{doc_id, shard_owner, metadata_holder, pdf_sha256, chunks_sha256}`.
     - `BACKUP_CHECK`: verifies off-node PDF backup replicas match the on-chain `pdf_sha256`.
     - `LORA_CLIENT_SUBMIT` & `LORA_ROUND_FINALIZE`: commits and verifies each peer's LoRA adapter SHA-256 before and after weighted `FedAvg` / `FFA-LoRA`.

2. **50/50 Document & PDF Sharding + Private Kubo IPFS (`fedrag/shard_store.py`, `fedrag/ipfs_client.py`, `connection-profile.json`)**
   - **Private Kubo Swarm (`LIBP2P_FORCE_PNET=1` + `swarm.key`):** `ipfs-node1` and `ipfs-node2` run with public bootstrap peers removed (`ipfs bootstrap rm --all`), forming an isolated consortium IPFS swarm.
   - **Node 1 (`shard-1`)** pins the first 50% of full PDFs to `ipfs-node1` (`POST /api/v0/add?pin=true&cid-version=1`), indexes their chunks + embeddings (`OllamaEmbedder` / `HashEmbedder`), and stores metadata-only stubs (`ipfs_cid` + verified SHA-256 hashes) and backup pins for the second 50%.
   - **Node 2 (`shard-2`)** pins the second 50% of full PDFs to `ipfs-node2`, indexes their chunks + embeddings, and stores metadata-only stubs and backup pins for the first 50%.
   - **Connection Profile (`connection-profile.json`):** Maps each organization/peer (`MaestroOrg`, `Peer2Org`) to its mTLS Raft Orderer, RAG/Trainer service, X.509 certificates, and local Kubo IPFS RPC endpoint.

3. **Federated LoRA Training (`fedrag/trainer.py`, `fedrag/fedavg.py`, `fedrag/hw_training/`)**
   - Supports both standard `fedavg` LoRA and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`, freezing deterministically seeded `lora_A` and training/averaging `lora_B`).
   - Full CUDA (`train-safe`) and Apple Silicon MPS (`torch.mps.set_per_process_memory_fraction` + RSS watchdog) trainers ported from `softone-rag` live in `fedrag/hw_training/`.

## Quick Start (`./run.sh` CLI)

You can run the full pipeline at once or run individual commands step-by-step:

```bash
# Run everything end-to-end (certs -> build -> up -> ingest -> train -> query -> tamper -> status)
./run.sh all

# Or run individual commands each time:
./run.sh certs                                           # Generate X.509 CA & node mTLS certificates
./run.sh build                                           # Build Docker images for node1-maestro & node2-peer
./run.sh up                                              # Start the 2 connected mTLS Docker containers
./run.sh ingest                                          # Ingest 50/50 sharded PDFs, SHA-256 stubs & backups
./run.sh train 3 fedavg                                  # Run 3 rounds of standard FedAvg-LoRA
./run.sh train 3 ffa                                     # Run 3 rounds of FFA-LoRA (Freeze-A, arXiv:2403.12313)
./run.sh query "Which SODTYPE code identifies customers?" # Decentralized scatter-gather RAG query
./run.sh tamper                                          # Test Raft rejection of a corrupted backup hash
./run.sh status                                          # Check 50/50 shard counts, backups & Raft height
./run.sh ledger                                          # Print the full Raft block ledger JSON
./run.sh down                                            # Stop and clean up containers & volumes
```

## Offline tests

```bash
PYTHONPATH=. uv run --with pytest --with numpy pytest -q
PYTHONPATH=. uv run --with ruff ruff check --config pyproject.toml fedrag scripts tests
node --experimental-strip-types chaincode/validator.test.ts   # TypeScript chaincode twin
```
