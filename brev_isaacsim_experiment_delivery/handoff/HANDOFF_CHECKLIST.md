# Handoff Checklist

This checklist enumerates the artifacts included in the `brev_isaacsim_experiment_delivery` package and provides instructions for the recipient to verify and use them.  Complete all items marked `[ ]` to ensure that the experiment can be reproduced without additional clarification.

## Directory Verification

- [ ] Confirm that the folder structure matches `PROJECT_STRUCTURE.txt`.
- [ ] Ensure that `setup/`, `src/`, `configs/`, `notebooks/`, `data_logs/`, `runs/`, `figures/`, `report/` and `handoff/` exist and are populated as described.
- [ ] Verify that `SYSTEM_VERSIONS.md` and `PAPER_UPDATE_NOTES.md` are filled out with the environment details and required paper changes.

## Environment Setup

- [ ] On a Brev instance, open a browser terminal and run `./setup/install.sh` to install the environment (requires sudo privileges).  Follow any prompts for NVIDIA driver installation.
- [ ] Activate the Python virtual environment (`source ~/isaacsim_env/bin/activate`) and ensure that `python --version` reports Python 3.11.
- [ ] Run `./setup/launch.sh` to start the docker-compose services.  Use `docker compose ps` to confirm that all services are running.

## Demonstration Collection

- [ ] Open JupyterLab via the Brev browser UI and run `notebooks/01_generate_demos.ipynb`.  This will collect demonstration trajectories (dummy in the scaffold; replace with real tele‑operation as needed).
- [ ] Verify that the demonstrations are saved under `runs/demonstrations/`.

## Policy Fine‑Tuning

- [ ] Run `notebooks/02_fine_tune_policy.ipynb` to fine‑tune the policy on the demonstrations.  Confirm that checkpoints are created in the `checkpoints/` directory.

## Calibration and Evaluation

- [ ] Run `notebooks/03_calibrate_eval.ipynb` to fit the temperature scaling calibrator and simulate evaluation episodes.  After completion, verify that log files exist in `data_logs/` (`episodes.csv`, `steps.csv`, `interventions.csv`, `contacts.csv`).
- [ ] Inspect the logs using pandas (e.g. `pd.read_csv('data_logs/episodes.csv')`) to ensure that they contain the expected columns and sample entries.

## Figure Generation

- [ ] Run `notebooks/04_generate_figures.ipynb` to generate Figures 6–10.  Check that the figure files appear in the `figures/` directory.
- [ ] Review the generated figures to confirm they are non‑empty and follow the intended layout.

## Report and Documentation

- [ ] Read `report/FINAL_IMPLEMENTATION_REPORT.md` to understand what was implemented, what remains to be done, and next steps for full reproduction.
- [ ] Use `PAPER_UPDATE_NOTES.md` to update the manuscript so it reflects the Brev‑compatible implementation.

## Packaging

- [ ] Create a ZIP archive of the entire `brev_isaacsim_experiment_delivery` directory (for example using `zip -r experiment_delivery.zip brev_isaacsim_experiment_delivery`).  Place the archive in `handoff/experiment_delivery.zip`.

Completing this checklist ensures that the recipient has a fully documented, reproducible scaffold for running the uncertainty‑calibrated safety gating experiment on NVIDIA Brev.  Any remaining work items (e.g. integrating Isaac Sim or the real policies) should be addressed before publishing results.