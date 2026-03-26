#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/isaac-lab"
docker compose up -d

echo "Launchable services started. Open your Brev secure link for VS Code, and use /viewer/ in a second tab for Isaac Sim streaming."
