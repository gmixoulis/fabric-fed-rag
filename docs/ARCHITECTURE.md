# Architecture

A decentralized, hash-sharded RAG and Federated LoRA training system modeled on
**Hyperledger Fabric 2.x Raft Ordering & Private Data Hash Commitments**. Nodes
communicate exclusively over **TLS 1.3 mutual authentication (mTLS)** using X.509
certificates issued by the consortium CA (`fedrag/certgen.py`).

The corpus is split into **N shards**, one per member of the consortium (N = 3 on this
cluster; N = 2 is the original 50/50 behaviour and still passes its own tests).
Membership comes from configuration, never from a hard-coded peer name.

## 1. Shard assignment and the backup ring (`fedrag/shard_store.py`)

* `ShardRing(members)` sorts the member ids; the position in that sorted list is the
  shard index (`shard-1`, `shard-2`, `shard-3`).
* `shard_index_for(doc_id, N, salt)` = `int(sha256("fabric-fed-rag/shard-ring/v1:" + doc_id), 16) % N`.
  Same document id, same salt, same N → same shard on every node, with no coordination.
* Every document has **exactly one owner** (the shard its hash selects) and **exactly one
  backup replica holder**, the next node on the ring: **1 backs up 2, 2 backs up 3,
  3 backs up 1**. With N = 2 each node backs up the other, which is the old behaviour.
* Every other member keeps a **metadata-only stub** (all SHA-256 hashes, no chunks).
  So every node can answer "does this document exist, and what are its committed hashes"
  for the whole corpus, while holding full text for its own shard plus one replica.
* Restore: `read_backup(doc_id)` re-hashes the replica and compares it with both the
  `BACKUP_CHECK` commitment and the manifest; `ingest_restored_backup()` re-extracts the
  chunks (`CHUNK[i]` markers in the PDF), re-embeds them and verifies the original hashes.
  `POST /shard/restore` on the owner pulls replicas from the ring successor over mTLS.

## 2. Three-node Docker cluster topology (WP3, one PC)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ node1-maestro — Raft Leader/Orderer (:7050) + RAG/Trainer (:8443)        │
│   ├─ ShardedRAGStore   shard-1: full PDFs + chunks + embeddings + pairs  │
│   ├─ Metadata stubs    shard-2, shard-3                                  │
│   ├─ Backup replica    shard-2   (ring: 1 backs up 2)                    │
│   ├─ ipfs-node1 (Kubo private swarm)   ollama-node1 (own embedding model)│
│   └─ Federated coordinator                                               │
├──────────────────────────────────────────────────────────────────────────┤
│ node2-peer — Raft Follower + RAG/Trainer                                 │
│   ├─ ShardedRAGStore   shard-2 …  stubs shard-1, shard-3  backup shard-3 │
│   └─ ipfs-node2                        ollama-node2                      │
├──────────────────────────────────────────────────────────────────────────┤
│ node3-peer — Raft Follower + RAG/Trainer                                 │
│   ├─ ShardedRAGStore   shard-3 …  stubs shard-1, shard-2  backup shard-1 │
│   └─ ipfs-node3                        ollama-node3                      │
└──────────────────────────────────────────────────────────────────────────┘
        all node-to-node traffic: TLS 1.3 mTLS, ssl.CERT_REQUIRED
```

Raft membership is `RAFT_MEMBER_URLS` (three members, quorum 2 of 3); the RAG/shard
membership is `FEDRAG_MEMBERS` (same node ids, `:8443` URLs). Both are parsed by
`parse_member_urls`, so `./run.sh status` reports one topology, not two.

`FEDRAG_MEMBERS` drives four things (`fedrag/node_service.py`):

| Behaviour | All peers alive | One peer down |
|---|---|---|
| Ingest stubs | stub written on every other member | ingest completes, `replica_warnings` names the peer |
| Backup replica | ring successor stores + attests (`BACKUP_CHECK`) | ingest completes with a warning naming the peer |
| Scatter-gather query | top-k over all shards | per-peer timeout, `missing_nodes` + `missing_shards` reported, `complete=false`, lower recall |
| Federated round | N participant states aggregated | trains on the reachable members; refused below the ledger quorum (2 of 3) |

Embedding-model agreement (WP2) is checked against every reachable member and against
every `DOC_MANIFEST` on the ledger before an ingest writes anything. A reachable peer
with a different digest fails the ingest; an unreachable peer is reported as
`unreachable` and does not block the ingest of this node's own shard.

## 3. Federated LoRA aggregation (`fedrag/fedavg.py`, `fedrag/node_service.py`)

`POST /fed/run-round` on the coordinator:

1. trains on its own shard (`LORA_CLIENT_SUBMIT` on the ledger),
2. asks every other reachable member to train over mTLS, with a per-peer timeout,
3. verifies each returned tensor set against the submitter's SHA-256 commitment,
4. refuses the round if fewer than `lora_quorum` members (default: majority of the
   configured ledger members, i.e. 2 of 3) took part,
5. aggregates and commits `LORA_ROUND_FINALIZE` with `participants`,
   `participant_hashes`, `unavailable_members`, `aggregation` and `server_lr`.

Two aggregation modes are supported:

| `aggregation` | Server learning rate | Merge |
|---|---|---|
| `average` (default) | 1 | weighted mean of the participant adapters (plain FedAvg, bit-identical to the two-node result) |
| `summed` | number of participants | `previous + N · (mean − previous)`, i.e. the per-node updates are summed, not halved (softone-rag E-2026-11-18 for disjoint shards) |

## 4. Multi-machine hardware topology (stage 2: Linux PC + 2× MacBook Air M4)

The third node is designed to move to the MacBook (WP5) with the same certificates, the
same `shard-3` and the same ledger rules: the node service runs natively there (MPS
trainer with the RSS watchdog) and its Ollama is already installed. Nothing in the shard
ring, the ledger or the aggregation assumes the three members share a host — only the
wall-clock measurements do (training is sequential on the single PC GPU).

## 5. On-chain vs. off-chain data separation

| Asset Type | On-Chain (Raft Ledger on All Nodes) | Off-Chain (Shard Owner Node) | Off-Chain (Stub Holders / Backup Node) |
|---|---|---|---|
| **PDF & Chunks** | `DOC_MANIFEST`: `doc_id`, `shard_owner`, `metadata_holder`, `shard_index`, `shard_count`, `backup_node`, `ipfs_cid`, `pdf_sha256`, `chunks_sha256`, `num_chunks`, `page_count`, `embedding_model_digest` | Full PDF binary (`pdfs/<doc_id>.pdf`), SQLite chunks, float64 embedding vectors | Metadata-only stub (`is_metadata_only = 1`) on every other member; one verified replica (`backups/<doc_id>.backup.pdf`) on the ring successor |
| **Backup Attestation** | `BACKUP_CHECK`: `doc_id`, `source_node`, `backup_node`, `computed_sha256`, `verified=true` | Source PDF | Recomputes `sha256(bytes)` and commits `BACKUP_CHECK` only if it matches `DOC_MANIFEST.pdf_sha256` |
| **Federated LoRA Round** | `LORA_CLIENT_SUBMIT` & `LORA_ROUND_FINALIZE`: `round_id`, `lora_mode`, `aggregation`, `server_lr`, `adapter_sha256`, `global_adapter_sha256`, `participants`, `unavailable_members`, `num_examples`, `weighted_loss` | Local shard training pairs + local `.npz`/`.safetensors` adapter | Coordinator verifies client hashes against `LORA_CLIENT_SUBMIT` before committing `LORA_ROUND_FINALIZE` |

## 6. Still two-node (deliberately, for now)

* `tests/test_raft_mtls_cluster.py` and `docs/REAL_EVAL_RESULTS.md`: the N = 2 regression
  and its recorded numbers. They are kept because WP3 must not change N = 2 behaviour.
* `scripts/run_real_eval.py` still exposes the mode key `decentralized_50_50_rag` (results
  files and the Excel exporter depend on it) although it now queries all three shards.
* `fedrag/hw_training/`: the CUDA/MPS trainers take one PC and one Mac shard; a third
  trainer host is WP5.
* The two-node research numbers in `docs/RESEARCH_PLAN.md` and `docs/EXPERIMENTS.md`
  (E1–E8) are historical and are not rewritten.
