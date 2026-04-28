#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source envp/bin/activate
./envp/bin/dvc repro
docker compose up --build -d

echo "API running at http://localhost:8001"
echo "Health check: curl http://localhost:8001/health"
