#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}
PROJECT_DIR=${PROJECT_DIR:-/workspace/brev_isaacsim_experiment_delivery}

if [ -z "${PASSWORD:-}" ]; then
    AUTH_MODE="none"
else
    AUTH_MODE="password"
fi

bootstrap_project() {
    local req_file="${PROJECT_DIR}/requirements.txt"
    local marker="${PROJECT_DIR}/.brev_bootstrap_complete"

    if [ -f "$req_file" ] && [ ! -f "$marker" ]; then
        echo "[entrypoint] Installing project Python dependencies from $req_file"
        if python -m pip install --upgrade pip && python -m pip install -r "$req_file"; then
            touch "$marker"
        else
            echo "[entrypoint] Warning: dependency bootstrap failed; starting code-server anyway." >&2
        fi
    fi
}

bootstrap_project

exec code-server --bind-addr=127.0.0.1:8080 --auth="${AUTH_MODE}" "${WORKSPACE_DIR}"
