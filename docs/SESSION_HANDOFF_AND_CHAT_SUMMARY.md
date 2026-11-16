# Session Handoff & Complete Conversation Summary (2026-11-16)

This document preserves the complete context, architectural decisions, literature review,
paper comparisons, debugging discoveries, and experimental results from the 2026-11-16
research and implementation session so you (or any coding agent tomorrow) can resume
immediately.

---

## 1. Original Research Vision & Architecture Agreed Upon

- **Goal:** Combine **`llama.cpp` / Ollama**, **Federated LoRA Training**, **Decentralized 50/50 Sharded PDF RAG with Embeddings**, **Private Kubo IPFS (`LIBP2P_FORCE_PNET=1`)**, and a **Hyperledger Fabric 2.x–style Raft + mTLS Ledger** across heterogeneous consumer hardware (Linux PC with GTX 1660 SUPER 6 GB CUDA + 2× MacBook Air M4 16 GB Metal/MPS).
- **Why 50/50 Sharding + Metadata Stub PDFs + Kubo IPFS:**
  - Putting raw PDFs, vector embeddings, or multi-megabyte LoRA weights inside blockchain blocks causes severe ledger bloat.
  - Instead:
    - **Node 1 (`node1-maestro` + `ipfs-node1`)** stores the **first 50% of full PDFs** (`pdfs/<doc_id>.pdf` with `content_sha256` and `chunks_sha256` stamped in the PDF `/Info` dictionary), pins them to its local Private Kubo IPFS node (`ipfs_cid`), and indexes their chunks + `embeddinggemma`/`bge-m3` embeddings in SQLite (`rag_shard.sqlite`). For the **second 50%**, Node 1 stores only a **Metadata-Only Stub PDF (`pdfs/<doc_id>.stub.pdf`)** carrying the verified `pdf_sha256`, `content_sha256`, `chunks_sha256`, and `ipfs_cid` (with its own `stub_pdf_sha256` validated) plus a verified backup replica (`backups/<doc_id>.backup.pdf`).
    - **Node 2 (`node2-peer` + `ipfs-node2`)** stores the exact inverse: the **second 50% of full PDFs + embeddings** and the **first 50% as Metadata-Only Stub PDFs + verified backups**.
    - **On-Chain Raft Ledger (`fedrag/raft_ledger.py` & `chaincode/validator.ts`):** Every node holds the identical hash-chained Raft ledger recording `DOC_MANIFEST`, `BACKUP_CHECK`, `LORA_CLIENT_SUBMIT`, and `LORA_ROUND_FINALIZE`.
- **Maestro Node Role:**
  - Runs the Raft Leader / Orderer, the X.509 mTLS Certificate Authority (`fedrag/certgen.py`), the Federated LoRA Coordinator (`FedAvg-LoRA` & `FFA-LoRA`), and the Decentralized Scatter-Gather RAG Router.

---

## 2. How This Work Differs from *Federated TrustChain* (`arXiv:2406.04076`, Zuo et al., 2024)

We analyzed the full text of *Federated TrustChain: Blockchain-Enhanced LLM Training and Unlearning* (`arXiv:2406.04076`) and established seven concrete differentiators for our paper:

| Dimension | *Federated TrustChain* (`arXiv:2406.04076`) | Our System (`fabric-fed-rag` + `softone-rag`) |
|---|---|---|
| **Primary Objective** | Machine Unlearning (forgetting $D_{\text{forget}}$ on sentiment classification) | **Decentralized 50/50 Sharded PDF RAG + Federated LoRA Training** |
| **RAG, Embeddings & PDFs** | **None** (no RAG, no embeddings, no PDFs) | **50/50 Sharded PDF RAG** (`PDFEmbedder`, `embeddinggemma`/`bge-m3`, metadata stub PDFs) |
| **Off-Chain Storage** | None (Algorithms 2, 3 & 5 upload $LLM_g$ directly to Smart Contracts) | **Private Kubo IPFS Swarm (`LIBP2P_FORCE_PNET=1`, `swarm.key`)** + CIDv1 pinning |
| **On-Chain Footprint** | Full model parameters & in-contract validation | **Lightweight 64-char SHA-256 + CIDv1 manifests only** (`DOC_MANIFEST`, `BACKUP_CHECK`, `LORA_CLIENT_SUBMIT`, `LORA_ROUND_FINALIZE`) |
| **Models & Hardware** | GPT-2 (124M, 2019) on 1× NVIDIA A6000 (48 GB) | **`Qwen3.5-4B` / `Gemma4-26B`** via `llama.cpp` + Ollama across heterogeneous consumer nodes (6 GB CUDA PC + 16 GB M4 Macs) |
| **LoRA Aggregation** | Single-node LoRA hyperparameter sweep $(r, \alpha, \text{dropout})$ | **Weighted `FedAvg-LoRA` vs. `FFA-LoRA` (`arXiv:2403.12313`)** with deterministic `lora_A` seeding & multimodal/text key normalization |
| **Transport Security** | Application-level JWT tokens | **Transport-level TLS 1.3 Mutual Authentication (mTLS)** with X.509 certificates (`ssl.CERT_REQUIRED`) |

---

## 3. Curated 2024–2026 Related Work Bibliography (12 Papers)

1. **Federated TrustChain: Blockchain-Enhanced LLM Training and Unlearning** — `arXiv:2406.04076` (IEEE, Jun 2024)
2. **FD-RAG: Federated Dual-System Retrieval-Augmented Generation** — `arXiv:2605.27432` (May 2026)
3. **A Decentralized RAG System with Source Reliabilities Secured on Blockchain** — `arXiv:2511.07577` (Nov 2025)
4. **DRAG: Distributed Retrieval-Augmented Generation** — `arXiv:2505.00443` (May 2025)
5. **FedE4RAG: Privacy-Preserving Federated Embedding Learning for Localized RAG** — `arXiv:2504.19101` (Apr 2025)
6. **C-FedRAG: A Confidential Federated Retrieval-Augmented Generation System** — `arXiv:2412.13163` (Dec 2024)
7. **TAD-LoRA: Topology-Aware Decentralized LoRA** — `arXiv:2602.00451` (Feb 2026)
8. **Dec-LoRA: Decentralized LoRA for Large Language Models** — `arXiv:2501.15361` (Jan 2025)
9. **FLoRA: Federated Fine-Tuning LLMs with Heterogeneous Low-Rank Adaptations** — `arXiv:2409.05976` (NeurIPS 2024)
10. **FFA-LoRA: Improving LoRA in Privacy-Preserving Federated Learning** — `arXiv:2403.12313` (ICLR 2024)
11. **FlexLoRA / HetLoRA: Federated Fine-Tuning of LLMs under Heterogeneous Resources** — `arXiv:2402.11505` (ACL/EMNLP 2024)
12. **HAFLQ & FAH-QLoRA: Heterogeneous Adaptive Federated LoRA with Quantization** — arXiv (2024–2025)

---

## 4. Key Engineering & Debugging Lessons Discovered Today

1. **Canonical JSON Before Hashing Raft Entries (`fedrag/raft_ledger.py`):**
   - Computing `entry_hash` before canonicalizing JSON (`json.dumps(payload, sort_keys=True, separators=(",", ":"))`) caused leader/follower hash mismatches over HTTP replication. Canonical JSON + a deterministic genesis timestamp (`2026-01-01T00:00:00Z`) fixed Raft quorum replication completely.
2. **RFC 5280 X.509 Extensions for Python 3.15+ (`fedrag/certgen.py`):**
   - Python 3.15's OpenSSL wrapper rejects CA/leaf certificates missing `subjectKeyIdentifier=hash` and `authorityKeyIdentifier=keyid,issuer`. Both are now generated automatically by `fedrag/certgen.py`.
3. **Kubo 0.43+ Private Network Requirement (`scripts/ipfs_init.sh`):**
   - `ipfs/kubo:latest` (v0.43.1) refuses to start when `swarm.key` / `LIBP2P_FORCE_PNET=1` is set unless `ipfs config --json AutoConf.Enabled false` is configured alongside `ipfs bootstrap rm --all`.

---

## 5. How to Resume Tomorrow

```bash
cd /Users/gmixoulis/Desktop/my-projects/fabric-fed-rag

# 1. Run fast offline unit & lint gates (9/9 pytest + ruff)
./run.sh unit

# 2. Run the 4-container mTLS Raft + Private Kubo IPFS + Sharded RAG + LoRA demo
./run.sh all

# 3. Run the full Real-Case Ollama Benchmark (embeddinggemma + qwen3.5:4b + 12 held-out cases)
./run.sh real-eval
```
