#!/bin/bash
set -e

python3 -m fedrag.raft_ledger \
  --node-id "${NODE_ID}" \
  --role "${RAFT_ROLE}" \
  --listen-port 7050 \
  --leader-url "${LEADER_URL}" \
  --peers "${RAFT_PEERS}" \
  --member-urls "${RAFT_MEMBER_URLS}" \
  --lora-quorum "${LORA_QUORUM:-2}" \
  --ca "/certs/ca.crt" \
  --cert "/certs/${NODE_ID}.crt" \
  --key "/certs/${NODE_ID}.key" \
  --ledger "/data/ledger.json" &

exec python3 -m fedrag.node_service \
  --node-id "${NODE_ID}" \
  --shard-id "${SHARD_ID}" \
  --listen-port 8443 \
  --data-dir "/data" \
  --raft-url "https://${NODE_ID}:7050" \
  --peer-url "${PEER_RAG_URL}" \
  --ca "/certs/ca.crt" \
  --cert "/certs/${NODE_ID}.crt" \
  --key "/certs/${NODE_ID}.key"
