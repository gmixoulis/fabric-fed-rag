# Architecture

A decentralized, 50/50 storage-sharded RAG and Federated LoRA training system
modeled on **Hyperledger Fabric 2.x Raft Ordering & Private Data Hash Commitments**.
Nodes communicate exclusively over **TLS 1.3 mutual authentication (mTLS)** using
X.509 certificates issued by the consortium CA (`fedrag/certgen.py`).

## 1. Two-Node Docker Cluster Topology (Stage 1 Demo)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Docker Container 1: node1-maestro (Maestro + Shard 1 Peer)              │
│  ├─ fedrag.raft_ledger (:7050, mTLS) ── Raft Leader & Chaincode Engine  │
│  └─ fedrag.node_service (:8443, mTLS)                                   │
│       ├─ ShardedRAGStore (Shard 1: 50% Full PDFs + vectors + QA pairs)  │
│       ├─ Metadata Stubs  (Shard 2: 50% metadata + verified SHA-256)     │
│       ├─ Backup Store    (Shard 2: verified PDF replicas)               │
│       └─ Federated Coordinator & Local LoRA Trainer (FedAvg / FFA-LoRA) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  TLS 1.3 mTLS (fabric-mtls-net)
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Docker Container 2: node2-peer (Follower + Shard 2 Peer)                │
│  ├─ fedrag.raft_ledger (:7050, mTLS) ── Raft Follower & Chaincode       │
│  └─ fedrag.node_service (:8443, mTLS)                                   │
│       ├─ ShardedRAGStore (Shard 2: 50% Full PDFs + vectors + QA pairs)  │
│       ├─ Metadata Stubs  (Shard 1: 50% metadata + verified SHA-256)     │
│       ├─ Backup Store    (Shard 1: verified PDF replicas)               │
│       └─ Local LoRA Trainer (FedAvg / FFA-LoRA)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Multi-Machine Hardware Topology (Stage 2: Linux PC + 2× MacBook Air M4)

```
Linux PC (GTX 1660 SUPER 6 GB, CUDA)
  ├─ Maestro Orderer & Coordinator
  ├─ Ollama embeddings (bge-m3 / embeddinggemma)
  ├─ llama-server (--lora global_adapter.gguf --rpc MAC1:50052)
  └─ fedrag/hw_training/client.py (via ~/.local/bin/train-safe, 4-bit QLoRA)
        │
        ├── mTLS over LAN / Tailscale ──► MacBook Air M4 #1 (16 GB Unified, Metal/MPS)
        │                                   ├─ Shard 1 RAG + PDF Store
        │                                   └─ client.py (bf16 MPS, 0.75 cap + RSS watchdog)
        │
        └── mTLS over LAN / Tailscale ──► MacBook Air M4 #2 (16 GB Unified, Metal/MPS)
                                            ├─ Shard 2 RAG + PDF Store
                                            └─ client.py (bf16 MPS, 0.75 cap + RSS watchdog)
```

## 3. On-Chain vs. Off-Chain Data Separation

| Asset Type | On-Chain (Raft Ledger on All Nodes) | Off-Chain (Shard Owner Node) | Off-Chain (Peer / Backup Node) |
|---|---|---|---|
| **PDF & Chunks** | `DOC_MANIFEST`: `doc_id`, `shard_owner`, `metadata_holder`, `pdf_sha256`, `chunks_sha256`, `num_chunks`, `page_count` | Full PDF binary (`pdfs/<doc_id>.pdf`), SQLite chunks, float64 embedding vectors | Metadata-only stub (`is_metadata_only = 1`) + verified backup copy (`backups/<doc_id>.backup.pdf`) |
| **Backup Attestation** | `BACKUP_CHECK`: `doc_id`, `source_node`, `backup_node`, `computed_sha256`, `verified=true` | Source PDF | Recomputes `sha256(bytes)` and commits `BACKUP_CHECK` only if it matches `DOC_MANIFEST.pdf_sha256` |
| **Federated LoRA Round** | `LORA_CLIENT_SUBMIT` & `LORA_ROUND_FINALIZE`: `round_id`, `lora_mode`, `adapter_sha256`, `global_adapter_sha256`, `num_examples`, `weighted_loss` | Local shard training pairs + local `.npz`/`.safetensors` adapter | Maestro verifies client hashes against `LORA_CLIENT_SUBMIT` before committing `LORA_ROUND_FINALIZE` |
