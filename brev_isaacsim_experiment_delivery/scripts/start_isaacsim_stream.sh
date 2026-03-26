#!/usr/bin/env bash
set -euo pipefail
export ACCEPT_EULA=${ACCEPT_EULA:-Y}

if [ ! -x /isaac-sim/runheadless.sh ]; then
  echo "Isaac Sim runtime not found. Run this script inside the launchable vscode container." >&2
  exit 1
fi

echo "Starting Isaac Sim headless with browser streaming enabled..."
echo "Open the same Brev URL in a second tab and append /viewer/ once the app is ready."
exec /isaac-sim/runheadless.sh
