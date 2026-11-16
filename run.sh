#!/bin/bash
# FabricFedRAG Cluster & Experiment CLI Runner
# Usage:
#   ./run.sh all                      # Full flow: certs -> build -> up -> ingest -> train -> query -> tamper -> status
#   ./run.sh certs                    # Generate Root CA & node X.509 mTLS certificates
#   ./run.sh build                    # Build Docker images for node1-maestro and node2-peer
#   ./run.sh up                       # Start the 2 connected mTLS Docker containers
#   ./run.sh down                     # Stop and remove containers + volumes
#   ./run.sh ingest                   # Ingest 50/50 sharded PDFs, SHA-256 stubs, and cross-node backups
#   ./run.sh train [rounds] [mode]    # Run Federated LoRA rounds (e.g. ./run.sh train 3 fedavg OR ./run.sh train 3 ffa)
#   ./run.sh query "<question>"       # Run decentralized scatter-gather RAG query across both shards
#   ./run.sh tamper                   # Test Raft chaincode rejection of a tampered backup hash
#   ./run.sh status                   # Show 50/50 shard counts, backup counts, and Raft block height
#   ./run.sh ledger                   # Dump full Raft block ledger JSON
#   ./run.sh logs                     # Tail logs from both Docker containers
#   ./run.sh unit                     # Run offline pytest and ruff checks

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if python3 -c "import numpy" >/dev/null 2>&1; then
  PY="python3"
elif command -v uv >/dev/null 2>&1; then
  PY="uv run --with numpy python"
else
  PY="python3"
fi

CMD="${1:-all}"

ensure_certs() {
  if [ ! -f "./certs/ca.crt" ] || [ ! -f "./certs/node1-maestro.crt" ] || [ ! -f "./certs/node2-peer.crt" ]; then
    echo ">>> Generating X.509 CA & mTLS certificates..."
    $PY -m fedrag.certgen --out ./certs --nodes node1-maestro,node2-peer
  fi
}

case "$CMD" in
  certs)
    $PY -m fedrag.certgen --out ./certs --nodes "${2:-node1-maestro,node2-peer}" --ips "${3:-127.0.0.1}"
    ;;

  build)
    ensure_certs
    docker compose build
    ;;

  up)
    ensure_certs
    docker compose up -d --build
    $PY scripts/run_demo.py --action health
    ;;

  down)
    docker compose down -v
    ;;

  ingest)
    $PY scripts/run_demo.py --action ingest
    ;;

  train)
    ROUNDS="${2:-3}"
    MODE="${3:-fedavg}"
    $PY scripts/run_demo.py --action train --rounds "$ROUNDS" --lora-mode "$MODE"
    ;;

  query)
    QUESTION="${2:-}"
    if [ -n "$QUESTION" ]; then
      $PY scripts/run_demo.py --action query --query "$QUESTION"
    else
      $PY scripts/run_demo.py --action query
    fi
    ;;

  tamper)
    $PY scripts/run_demo.py --action tamper
    ;;

  status)
    $PY scripts/run_demo.py --action status
    ;;

  ledger)
    $PY scripts/run_demo.py --action ledger
    ;;

  logs)
    docker compose logs -f
    ;;

  unit)
    PYTHONPATH=. uv run --with pytest --with numpy pytest -q
    ;;

  real-eval)
    ensure_certs
    docker compose down -v >/dev/null 2>&1 || true
    FEDRAG_EMBEDDER=ollama docker compose up -d --build
    $PY scripts/run_real_eval.py
    ;;

  all)
    $PY -m fedrag.certgen --out ./certs --nodes node1-maestro,node2-peer
    docker compose down -v >/dev/null 2>&1 || true
    docker compose up -d --build
    $PY scripts/run_demo.py --action all
    ;;

  *)
    echo "Unknown command: $CMD"
    echo "Available commands: all | certs | build | up | down | ingest | train [rounds] [fedavg|ffa] | query [question] | tamper | status | ledger | logs | unit"
    exit 1
    ;;
esac
