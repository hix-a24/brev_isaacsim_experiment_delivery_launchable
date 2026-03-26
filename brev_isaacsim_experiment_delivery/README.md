# brev_isaacsim_experiment_delivery

This workspace now includes a runnable synthetic experiment stack designed for Brev delivery when the full Isaac Sim runtime and proprietary policy checkpoints are not present.

## One-command end-to-end run

From the VS Code terminal inside the launchable:

```bash
cd /workspace/brev_isaacsim_experiment_delivery
./scripts/bootstrap_project.sh
./scripts/run_end_to_end.sh
```

That command sequence will:

- train lightweight synthetic OpenVLA and Octo checkpoint files into `checkpoints/`
- execute gated evaluation rollouts across both tasks and all configured shift axes
- write CSV logs into `data_logs/`
- render figures into `figures/`
- write a run manifest into `runs/latest/`

## Fast validation run

```bash
python scripts/run_smoke_test.py
```

## Isaac Sim streaming

To launch the browser stream from the same Brev workspace:

```bash
cd /workspace/brev_isaacsim_experiment_delivery
./scripts/start_isaacsim_stream.sh
```

Then open the same Brev URL with `/viewer/` appended.

## Scope

This repository now executes end to end without external robotics dependencies by using a deterministic synthetic task backend that mirrors the same control flow as the intended experiment stack:

- task reset and stepped rollouts
- policy checkpoint loading
- calibration and risk estimation
- hysteresis gating and planner fallback
- logging and post-run figure generation

When you later replace the synthetic task backend with real Isaac Sim scenes and real model checkpoints, the pipeline entrypoints and artifact generation paths can remain the same.
