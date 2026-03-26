# Brev Launchable for `brev_isaacsim_experiment_delivery`

This repository packages your project as an NVIDIA Brev-friendly Isaac Lab launchable modeled on the upstream `isaac-launchable` template, and the included project workspace now runs an end-to-end synthetic experiment pipeline out of the box.

## What is included

- a browser-accessible code-server / VS Code container
- an nginx reverse proxy that exposes VS Code and the Omniverse web viewer on the same secure link
- the Omniverse Kit App Streaming web client under `/viewer/`
- your `brev_isaacsim_experiment_delivery` project mounted into the workspace at `/workspace/brev_isaacsim_experiment_delivery`
- helper scripts to bootstrap Python dependencies, launch Isaac Sim streaming, train synthetic checkpoints, run gated evaluation rollouts, and render figures

## Brev setup

Use this repository the same way the upstream Isaac Launchable template is used:

1. Create a new Brev Launchable.
2. Choose **VM Mode - Basic VM with Python installed**.
3. Use this setup script:

```bash
#!/bin/bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/brev_isaacsim_experiment_delivery_launchable}"
cd "$REPO_DIR/isaac-lab"
docker compose up -d
```

4. Choose **No** for the Jupyter Notebook experience.
5. Add a **Secure Link** on port **80**.
6. Open TCP/UDP ports **1024**, **47998**, and **49100** for Kit App Streaming.
7. Use an RTX-capable GPU instance.

## First use inside the launchable

Open the secure link, then in the VS Code terminal run:

```bash
cd /workspace/brev_isaacsim_experiment_delivery
./scripts/bootstrap_project.sh
./scripts/run_end_to_end.sh
```

That command sequence will generate:

- checkpoints in `checkpoints/`
- episode, step, intervention, and contact logs in `data_logs/`
- figures in `figures/`
- a run manifest in `runs/latest/`

For a shorter validation run:

```bash
python scripts/run_smoke_test.py
```

To launch the raw Isaac Sim stream from the same workspace:

```bash
cd /workspace/brev_isaacsim_experiment_delivery
./scripts/start_isaacsim_stream.sh
```

Then open a second browser tab at the same base URL with `/viewer/` appended.

## Current implementation boundary

The package now executes end to end without external robotics dependencies by using a deterministic synthetic backend for the two tasks, policy checkpoints, calibration, gating, planner fallback, and artifact generation.

This means the zip is runnable immediately on Brev for validation and delivery. Replacing the synthetic backend with real Isaac Sim scenes, real OpenVLA or Octo checkpoints, and a real MoveIt integration is still future work, but the pipeline and artifact paths are already in place.
