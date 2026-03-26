#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[install.sh] Bootstrapping Python environment."
"$ROOT_DIR/scripts/bootstrap_project.sh"
echo "[install.sh] To execute the full experiment stack, run:"
echo "  cd $ROOT_DIR && ./scripts/run_end_to_end.sh"
