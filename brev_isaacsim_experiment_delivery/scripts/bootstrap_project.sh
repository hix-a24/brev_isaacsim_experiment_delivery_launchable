#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"
printf 'PYTHONPATH=%s
' "$ROOT_DIR" > "$ROOT_DIR/.env"
touch "$ROOT_DIR/.brev_bootstrap_complete"
echo "Project bootstrap complete. PYTHONPATH set to $ROOT_DIR"
