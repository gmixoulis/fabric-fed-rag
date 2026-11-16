---
title: Raft JSON canonicalization, RFC 5280 mTLS certs, and FFA-LoRA on 50/50 shards
author: gmixoulis
date: 2026-11-16
id: l-fabfed01
scope: repo
status: active
src: agent
touches:
  - fedrag/raft_ledger.py
  - fedrag/certgen.py
  - fedrag/trainer.py
  - fedrag/node_service.py
---

## Decided
- Use canonical JSON (`json.dumps(payload, sort_keys=True, separators=(",", ":"))`) before computing `entry_hash` in `fedrag/raft_ledger.py` and initialize the genesis block with a deterministic timestamp (`2026-01-01T00:00:00Z`) so Leader and Follower nodes compute identical block hashes across network hops.
- Include `subjectKeyIdentifier=hash` and `authorityKeyIdentifier=keyid,issuer` in `fedrag/certgen.py` so X.509 mTLS certificates pass strict RFC 5280 validation on Python 3.15+ as well as Python 3.12 in Docker.
- Support both `fedavg` (updating both `lora_A` and `lora_B`) and `ffa` (*FFA-LoRA*, `arXiv:2403.12313`, freezing deterministically seeded `lora_A` and updating/averaging only `lora_B`) in `fedrag/trainer.py`.

## Tried and ruled out
- Hashing raw uncompacted JSON payloads before HTTP replication failed on followers because JSON re-serialization changed whitespace, causing `entry_hash` mismatches (`raft quorum not reached`).
- Generating minimal OpenSSL certificates without `authorityKeyIdentifier` worked on Python 3.12 inside Debian Bookworm containers but failed on the macOS host's Python 3.15 with `SSLCertVerificationError: Missing Authority Key Identifier`.

## Watch out
- Debian `/bin/sh` (`dash`) in `python:3.12-slim-bookworm` does not support `wait -n`; use `#!/bin/bash` and `exec` in `scripts/entrypoint.sh`.
