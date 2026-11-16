# Research Plan & Literature Review: Decentralized 50/50 Sharded RAG & Federated LoRA over Raft + mTLS

## 1. Research Objective

Design, implement, and benchmark a decentralized edge-LLM architecture combining:
1. **Hyperledger Fabric 2.x–style Raft Ordering & mTLS Control Plane**: A Maestro node (Orderer + Coordinator) and Peer nodes connected via mutual TLS (X.509 certificates) with minimal TypeScript/Python chaincode (`chaincode/validator.ts`, `fedrag/raft_ledger.py`) recording cryptographic SHA-256 commitments.
2. **Symmetric 50/50 Document & Vector Sharding with Metadata Hash Stubs**:
   - Peer Group A holds the full PDFs, extracted chunks, and `bge-m3`/`embeddinggemma` embeddings (via Ollama) for the first 50% of the corpus, plus metadata-only stubs carrying verified SHA-256 hashes for the second 50%.
   - Peer Group B holds the full PDFs, extracted chunks, and embeddings for the second 50%, plus metadata-only stubs for the first 50%.
   - Cross-hardware backup replicas are periodically verified against on-chain SHA-256 hashes (`BACKUP_CHECK`).
3. **Shard-Aligned Federated LoRA Training (`FedAvg-LoRA` vs. `FFA-LoRA`) & `llama.cpp` GGUF Serving**:
   - Each node trains a LoRA adapter on its local 50% shard across heterogeneous hardware (Linux PC GTX 1660 SUPER 4-bit CUDA via `train-safe` and 2× MacBook Air M4 bf16 MPS with memory-fraction cap and RSS watchdog).
   - The Maestro verifies client adapter SHA-256 hashes against the Raft ledger, aggregates weights via weighted `FedAvg`, converts the adapter to GGUF (`convert_lora_to_gguf.py`), and serves it with `llama-server --lora`.

---

## 2. Related Work & Literature Review (2024–2026)

### A. Blockchain-Secured Federated LLM Fine-Tuning
- **Federated TrustChain: Blockchain-Enhanced LLM Training and Unlearning** (`arXiv:2406.04076`, IEEE 2024): Integrates Hyperledger Fabric with LoRA fine-tuning to record tamper-proof model updates and unlearning proofs on private organizational datasets. *Limitation:* Does not address decentralized RAG storage sharding, document/chunk hash verification, or `llama.cpp` edge GGUF deployment.

### B. Decentralized & Federated RAG with Blockchain / Sharding
- **A Decentralized RAG System with Source Reliabilities Secured on Blockchain** (`arXiv:2511.07577`, Nov 2025): Uses blockchain smart contracts to track source reliability across decentralized RAG nodes.
- **D-RAG: A Privacy-Preserving Framework for Decentralized RAG Using Blockchain** (2025): Peer-to-peer context verification across multi-owner knowledge bases.
- **DRAG: Distributed Retrieval-Augmented Generation** (`arXiv:2505.00443`, May 2025): Peer-to-peer edge RAG using Topic-Aware Random Walks without a central vector repository.
- **C-FedRAG: A Confidential Federated RAG System** (`arXiv:2412.13163`, Dec 2024): Combines NVIDIA FLARE with confidential computing enclaves for cross-silo RAG.
- **FedE4RAG** (`arXiv:2504.19101`, Apr 2025) & **FD-RAG** (`arXiv:2605.27432`, May 2026): Federated embedding learning (`FedE4RAG`) and dual-system hypergraph memory over fragmented edge corpora (`FD-RAG`).

### C. Federated LoRA Across Heterogeneous Edge Hardware & Quantization
- **FFA-LoRA: Improving LoRA in Privacy-Preserving Federated Learning** (`arXiv:2403.12313`, ICLR 2024): Demonstrates that averaging both $A$ and $B$ in LoRA introduces cross-term interference ($\bar{B}\bar{A} \neq \frac{1}{N}\sum B_i A_i$); freezing deterministically seeded $A$ (`lora_A`) and training/averaging only $B$ (`lora_B`) removes aggregation noise and halves communication cost.
- **FlexLoRA / HetLoRA** (`arXiv:2402.11505`, ACL/EMNLP 2024) & **FLoRA** (`arXiv:2409.05976`, NeurIPS 2024): Aggregation schemes for clients with heterogeneous LoRA ranks and memory budgets.
- **FAH-QLoRA** & **HAFLQ** (2024–2025): Federated QLoRA across heterogeneously quantized base models (e.g., 4-bit NF4 on one node and bf16 on another).

---

## 3. Novelty & Experimental Hypotheses

1. **H1 (Storage & Integrity):** A 50/50 document/embedding split with on-chain SHA-256 metadata stubs reduces active vector/PDF footprint per peer by ~50% while achieving 100% global top-$k$ recall and 100% deterministic detection of corrupted PDF backups or poisoned LoRA adapters.
2. **H2 (FedAvg-LoRA vs. FFA-LoRA on Sharded Corpora):** Under non-IID 50/50 document sharding across heterogeneous quantization backends (4-bit CUDA + bf16 MPS), freezing deterministically seeded `lora_A` (`FFA-LoRA`) stabilizes multi-round convergence compared to full `FedAvg-LoRA`.
3. **H3 (End-to-End Edge Feasibility):** The entire Raft + mTLS + Sharded RAG + Federated LoRA pipeline operates within strict consumer memory caps (6 GiB CUDA cgroup on Linux PC, 14 GiB Metal cap on M4 MacBooks).
