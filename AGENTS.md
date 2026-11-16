# Instructions for coding agents

Read `docs/ARCHITECTURE.md` and `docs/RESEARCH_PLAN.md` first. Read `.trail/learnings/`
before touching `fedrag/` or `fedrag/hw_training/`. Reasons for past decisions and
measurements are in `docs/RESEARCH-LOG.md` and `docs/EXPERIMENTS.md`.

## Gates (run all before reporting done)

```bash
PYTHONPATH=. uv run --with pytest --with numpy pytest -q
PYTHONPATH=. uv run --with ruff ruff check --config pyproject.toml fedrag scripts tests
```

Unit tests are offline: deterministic `HashEmbedder`, temporary SQLite databases,
and ephemeral localhost TLS 1.3 sockets. No unit test requires an external network,
a GPU, or real client data.

## Rules that are not negotiable

1. **Client data and private keys never enter the repository.** Not snapshots, not
   `data/`, not `certs/*.key`, and not raw business rows in reports or test fixtures.
2. **All node-to-node traffic uses mutual TLS (TLS 1.3, `ssl.CERT_REQUIRED`).**
   Never downgrade `verify_mode` or accept unverified client certificates.
3. **Never store raw PDFs, full chunks, or LoRA weight blobs inside Raft ledger blocks.**
   Only metadata and 64-character SHA-256 commitments (`DOC_MANIFEST`, `BACKUP_CHECK`,
   `LORA_CLIENT_SUBMIT`, `LORA_ROUND_FINALIZE`) belong on the shared Raft ledger.
4. **Canonical JSON before hashing.** Always serialize transaction payloads with
   `json.dumps(payload, sort_keys=True, separators=(",", ":"))` before computing
   `entry_hash` so Leader and Follower hashes never diverge across HTTP/JSON hops.
5. **RFC 5280 X.509 extensions are mandatory.** Every generated CA and node certificate
   in `fedrag/certgen.py` must include `subjectKeyIdentifier=hash` and
   `authorityKeyIdentifier=keyid,issuer` so Python 3.15+ OpenSSL verification succeeds.
6. **Hardware training runs only through `~/.local/bin/train-safe` on the Linux PC**
   (finite systemd cgroup) and with `torch.mps.set_per_process_memory_fraction` + the
   RSS watchdog (`mac_run.sh`) on the MacBook Air M4s. Never raise the memory limits.
7. **Two Python environments for full LLM checkpoints.** App/node environment: lightweight
   `numpy` + stdlib. Hardware training venv (`fedrag/hw_training/`): `torch`,
   `transformers`, `peft`, `safetensors`.

## Layout

| Path | Contents |
|---|---|
| `fedrag/raft_ledger.py` | Python Raft Leader/Follower replication over TLS 1.3 mTLS & hash-chained ledger |
| `chaincode/validator.ts` | TypeScript chaincode validator for shard manifests, backups, and LoRA rounds |
| `fedrag/certgen.py` | OpenSSL X.509 CA + node mTLS certificate generator (with SANs & RFC 5280 SKI/AKI) |
| `fedrag/shard_store.py` | 50/50 PDF + vector sharding, metadata-only SHA-256 stubs, and backup checksum verifier |
| `fedrag/embed.py` | `OllamaEmbedder` (`bge-m3` / `embeddinggemma`) + deterministic `HashEmbedder` + cosine `top_k` |
| `fedrag/trainer.py` | Local shard LoRA trainer supporting both `fedavg` and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`) |
| `fedrag/fedavg.py` | LoRA key normalisation, shape checks, weighted FedAvg, and deterministic SHA-256 tensor hashing |
| `fedrag/node_service.py` | Per-node mTLS daemon (`:8443`) for sharded RAG, backup attestation, and Federated LoRA rounds |
| `fedrag/hw_training/` | Full CUDA (`train-safe`) & Apple Silicon MPS QLoRA/bf16 trainers ported from `softone-rag` |
| `scripts/run_demo.py` | Modular CLI & end-to-end verifier for the 2-Docker-node mTLS cluster |
| `run.sh` | Top-level CLI runner (`all`, `certs`, `build`, `up`, `down`, `ingest`, `train`, `query`, `tamper`, `status`, `ledger`, `unit`) |
| `docs/` | `ARCHITECTURE.md`, `RESEARCH_PLAN.md`, `RESEARCH-LOG.md`, `EXPERIMENTS.md` |
| `.trail/learnings/` | Persistent engineering learnings across sessions |

## Useful commands

```bash
./run.sh all                                      # full 2-Docker mTLS build, shard ingest, LoRA train, RAG query & audit
./run.sh train 3 fedavg                           # run 3 rounds of standard FedAvg-LoRA
./run.sh train 3 ffa                              # run 3 rounds of FFA-LoRA (Freeze-A, arXiv:2403.12313)
./run.sh query "Inventory goods SODTYPE in MTRL"  # scatter-gather RAG query across both shards
./run.sh unit                                     # run offline pytest suite
```

## Working style expected here

Smallest complete change. Reuse the layer that owns the behaviour. Report what
you did not verify. When a measurement fails, find out whether the system, the
test or the infrastructure is at fault before changing any of them.
