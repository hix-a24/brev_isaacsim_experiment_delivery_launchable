#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
LAUNCHABLE_DIR="$REPO_ROOT/isaac-lab"

if [ -d "$LAUNCHABLE_DIR" ]; then
  cd "$LAUNCHABLE_DIR"
  docker compose up -d
  echo "[launch.sh] Brev launchable services started."
fi

echo "[launch.sh] To run the experiment stack inside the workspace:"
echo "  cd $PROJECT_DIR && ./scripts/run_end_to_end.sh"
