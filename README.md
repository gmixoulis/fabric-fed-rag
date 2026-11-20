# FabricFedRAG (`fabric-fed-rag`)

Decentralized **hash-sharded PDF RAG** (one shard per organisation, N = 3 on this cluster)
and **Federated LoRA Training (`FedAvg` & `FFA-LoRA`, plain averaging or summed updates)**
coordinated over a **Python / TypeScript Raft + mutual TLS (mTLS)** ledger (modeled on
Hyperledger Fabric 2.x Raft Orderer + private data hash commitments).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the three-node topologies, the shard
ring and the aggregation rules, and [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the
research plan, hypotheses and 2024–2026 literature review.

## Architecture

1. **Raft + mTLS Ledger & TypeScript/Python Chaincode (`fedrag/raft_ledger.py`, `chaincode/validator.ts`, `fedrag/certgen.py`)**
   - Every node authenticates with X.509 certificates issued by the consortium CA (`fedrag/certgen.py`) over TLS 1.3 (`ssl.CERT_REQUIRED`).
   - Three organisations (`MaestroOrg`, `Peer2Org`, `Peer3Org`); Raft membership and election need 2 of 3.
   - `node1-maestro` runs the Raft Leader / Orderer and Federated Coordinator; `node2-peer` and `node3-peer` are Followers with their own RAG/Trainer service.
   - Minimal chaincode state validation enforces:
     - `DOC_MANIFEST`: commits `{doc_id, shard_owner, metadata_holder, shard_index, shard_count, pdf_sha256, chunks_sha256, embedding_model_digest}`.
     - `BACKUP_CHECK`: verifies one off-node PDF backup replica per document against the on-chain `pdf_sha256`.
     - `LORA_CLIENT_SUBMIT` & `LORA_ROUND_FINALIZE`: commits and verifies each member's LoRA adapter SHA-256 before and after aggregation; the finalize quorum is a parameter (2 of 3), not a constant.

2. **Hash shards, metadata stubs and one ring backup (`fedrag/shard_store.py`, `fedrag/ipfs_client.py`, `connection-profile.json`)**
   - **Shard ring:** `ShardRing(members)` sorts the member ids; `shard_index_for(doc_id, N)` = seeded SHA-256 of `doc_id` mod N, so every node computes the same owner without coordination. N = 2 reproduces the original 50/50 split and is still tested.
   - **Ownership:** each node pins and indexes the full PDFs, chunks, embeddings and training pairs of its own shard in its own Kubo node (`POST /api/v0/add?pin=true&cid-version=1`).
   - **Stubs:** every other member keeps a metadata-only stub carrying the verified SHA-256 hashes.
   - **Backups:** each node also holds ONE verified replica of the next node's shard (1 backs up 2, 2 backs up 3, 3 backs up 1), so one node can be lost; `POST /shard/restore` rebuilds a shard from those replicas.
   - **Private Kubo Swarm (`LIBP2P_FORCE_PNET=1` + `swarm.key`):** `ipfs-node1..3` run with public bootstrap peers removed, forming an isolated consortium IPFS swarm.
   - **Connection Profile (`connection-profile.json`):** maps each organisation/peer to its mTLS Raft orderer, RAG/Trainer service, X.509 certificates, and local Kubo IPFS RPC endpoint.

3. **Three-peer retrieval and federated rounds (`fedrag/node_service.py`, `fedrag/fedavg.py`)**
   - The member list (`FEDRAG_MEMBERS`) replaces the old single `peer_url`: ingest sends stubs to every other member and the backup replica to the ring successor; no peer name is hard-coded.
   - Scatter-gather query fans out to all peers with a per-peer timeout. A missing node lowers recall but never fails the query; the answer reports `missing_nodes`, `missing_shards` and `complete=false`.
   - `POST /fed/run-round` trains on every reachable member, verifies each adapter hash, aggregates N states and finalizes with the ledger quorum (2 of 3). `aggregation=average` (default, plain FedAvg) or `aggregation=summed` (server learning rate = number of participants, for disjoint shards).

4. **Federated LoRA Training (`fedrag/trainer.py`, `fedrag/fedavg.py`, `fedrag/hw_training/`)**
   - Supports both standard `fedavg` LoRA and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`, freezing deterministically seeded `lora_A` and training/averaging `lora_B`).
   - Full CUDA (`train-safe`) and Apple Silicon MPS (`torch.mps.set_per_process_memory_fraction` + RSS watchdog) trainers ported from `softone-rag` live in `fedrag/hw_training/`.

## Quick Start (`./run.sh` CLI)

You can run the full pipeline at once or run individual commands step-by-step:

```bash
# Run everything end-to-end (certs -> build -> up -> ingest -> train -> query -> tamper -> status)
./run.sh all

# Or run individual commands each time:
./run.sh certs                                           # Generate X.509 CA & node mTLS certificates (node1..3)
./run.sh build                                           # Build Docker images for node1-maestro, node2-peer, node3-peer
./run.sh up                                              # Start the 3 node + 3 ipfs + 3 ollama containers
./run.sh ingest                                          # Ingest hash-sharded PDFs, SHA-256 stubs & ring backups
./run.sh train 3 fedavg                                  # 3 rounds of plain FedAvg-LoRA over 3 nodes
./run.sh train 3 fedavg summed                           # 3 rounds with summed updates (server lr = participants)
./run.sh train 3 ffa                                     # 3 rounds of FFA-LoRA (Freeze-A, arXiv:2403.12313)
./run.sh query "Which SODTYPE code identifies customers?" # Decentralized scatter-gather RAG query
./run.sh tamper                                          # Test Raft rejection of a corrupted backup hash
./run.sh status                                          # Three shard owners, stub and backup counts, Raft height
./run.sh ledger                                          # Print the full Raft block ledger JSON
./run.sh down                                            # Stop and clean up containers & volumes
```

Memory: three Ollama containers at `mem_limit: 2g` plus the model on disk need about 5 GB
free; `./run.sh up` refuses to start below that and `./run.sh ollama-check` verifies each
node's embedding model and latency.

## Offline tests

```bash
PYTHONPATH=. uv run --with pytest --with numpy pytest -q
PYTHONPATH=. uv run --with ruff ruff check --config pyproject.toml fedrag scripts tests
node --experimental-strip-types chaincode/validator.test.ts   # TypeScript chaincode twin
docker compose config -q                                      # compose file is valid (no containers started)
```
