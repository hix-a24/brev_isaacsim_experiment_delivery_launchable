# Uncertainty-Calibrated Safety Gating Experiment on NVIDIA Brev

This repository contains a Brev-compatible skeleton for reproducing the experiments described in the paper “Uncertainty‑Calibrated Safety Gating for Vision‑Language‑Action Policies in Long‑Horizon Robotic Manipulation Under Domain Shift (Sim‑to‑Real)”.  It is designed to be run entirely from a web browser using an NVIDIA Brev Launchable and includes scripts, configuration files and logging schemas needed to implement the full simulation pipeline on top of Isaac Sim 5.x and ROS 2.

## What was implemented

* Two manipulation tasks (Task A – drawer‐object and Task B – cluttered pick‑and‑sort) are modelled in Isaac Sim using USD scene files and randomization scripts for lighting, textures, occlusion/clutter and sensor noise【725427445771147†L590-L684】.  The tasks follow the paper definitions: Task A requires opening a drawer, grasping a coloured block and placing it in a bin, then closing the drawer; Task B requires finding an object among clutter and sorting objects into left/right trays【725427445771147†L590-L622】.
* A policy runner wrapper supports both OpenVLA and Octo policies.  Fine‑tuning from the paper’s demonstration counts (50 demos for Task A, 70 for Task B) is parameterized via YAML configuration【725427445771147†L1230-L1267】.
* A safety supervisor implements the calibrated risk gating mechanism with hysteresis thresholds δ_low=0.2 and δ_high=0.5【725427445771147†L470-L490】.  When the calibrated risk estimate is above the high threshold, control is handed off to a fallback planner; if it falls between thresholds, the policy is paused and a re‑observation is taken.  The fallback planner uses MoveIt 2 to generate safe trajectories and completes sub‑tasks such as retreating to a safe pose, re‑orienting for better visibility or executing the remainder of the task【725427445771147†L470-L522】.
* Logging utilities define the episode‑level, step‑level, intervention and contact schemas needed to regenerate Figures 6–10.  Real data from simulator runs should be stored in the `data_logs/` directory and processed by the figure generation scripts.

## What was adapted from the paper

The original paper was written for Isaac Sim 5.0 on Python 3.8 and ROS 2 Foxy.  However, the official Isaac Sim 5.x documentation states that all Isaac Sim components require Python 3.11 and that the built‑in ROS 2 bridge is compatible with the Humble and Jazzy distributions【610528769204527†L425-L430】【860636887936188†L436-L460】.  For Brev compatibility, this project assumes:

* **Python 3.11**, CUDA 12.x, PyTorch 2.7 and TensorFlow 2.15.
* **ROS 2 Humble** (recommended for Ubuntu 22.04) compiled with Python 3.11 and loaded via Isaac Sim’s internal bridge【860636887936188†L436-L460】.
* **Isaac Sim 5.0.0** or newer installed via pip packages (`isaacsim[all,extscache]`) within a virtual environment【610528769204527†L425-L430】.
* An **RTX‑class GPU** (e.g. NVIDIA L40S) since Isaac Sim does not support non‑RTX GPUs.
* **Headless simulation** accessed through a browser‑based JupyterLab on the Brev instance, avoiding any native desktop streaming.

Other deviations from the paper are recorded in `PAPER_UPDATE_NOTES.md`.

## Browser workflow

To reproduce the experiments from a web browser you should:

1. **Deploy a Brev Launchable** configured with Ubuntu 22.04, a single NVIDIA L40S GPU and at least 50 GB of disk space.  Ensure that ports for JupyterLab (e.g. 8888) and any service API are open.
2. **Log in to Brev** using your NVIDIA account.  When prompted for credentials, complete the sign‑in in the browser; do not provide secrets via chat.
3. **Open the web‑based VS Code or terminal** provided by Brev.  Clone this repository into the home directory or attach it as a persistent volume.
4. **Run `./setup/install.sh`** to create the Python 3.11 environment and install Isaac Sim, ROS 2 Humble, OpenVLA dependencies and all Python requirements.  The script will also download assets and convert USD scene files if necessary.
5. **Start the services** using `docker compose up` or by running `./setup/launch.sh`.  This will launch separate containers for the simulator, policy runner, safety supervisor, planner and analysis service.
6. **Open JupyterLab** in the browser and run the notebooks in `notebooks/` to generate demonstrations, fine‑tune the policies, calibrate the risk model, evaluate each method under domain shifts and produce the final figures.  Use the provided YAML configs (`configs/`) to select between pilot and full‑scale runs.
7. **Inspect results**.  Intermediate and final results (logs, videos, metrics and figures) will be stored under `runs/`.  The figure generation script reads from these logs and produces publication‑ready figures saved in `figures/`.

The README will be updated as the implementation evolves.  See `report/FINAL_IMPLEMENTATION_REPORT.md` for a summary of what was executed and what remains to be done.