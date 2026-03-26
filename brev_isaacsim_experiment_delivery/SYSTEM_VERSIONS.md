# System Versions and Environment Details

This document records the exact software and hardware versions used to run the uncertainty‑calibrated safety gating experiments on NVIDIA Brev.  Keeping track of these details is critical for reproducibility.  Update this file after provisioning your Brev instance and installing all dependencies.

## Brev Setup

- **Launchable type**: (e.g. *VM mode* with one NVIDIA L40S GPU).  See the Brev instance dashboard for the exact configuration.
- **Region**: (e.g. `us-west-2`).
- **Instance start date**: Fill in the date when the Brev instance was launched.
- **Budget notes**: Record your initial credit balance and the approximate consumption for each stage of the experiment.

## Host Operating System

- **Distribution**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Kernel**: Provide the `uname -r` output.

## GPU and Drivers

- **GPU model**: NVIDIA RTX L40S (other RTX‑class GPUs may be used but must support hardware ray tracing and CUDA).
- **Driver version**: Document the installed NVIDIA driver version (e.g. 545.29).  Use `nvidia-smi` to verify.
- **CUDA version**: The CUDA toolkit version installed on the system (e.g. CUDA 12.3).

## Python Environment

- **Python version**: 3.11.x.  Isaac Sim 5.x requires Python 3.11【610528769204527†L425-L430】.  Record the exact patch version (e.g. 3.11.4).
- **Package manager**: `pip` (inside a virtual environment) or `conda` if preferred.
- **Virtual environment**: Note the path and creation command for the virtual environment (e.g. `python3.11 -m venv isaacsim_env`).

## Major Package Versions

Record the versions of the key packages used.  Some values below are examples; update them to match your environment.

| Package                          | Version             | Notes |
|---------------------------------|---------------------|-------|
| **Isaac Sim**                   | 5.0.0 or >=5.0.0    | Installed via pip (`isaacsim[all,extscache]`). |
| **ROS 2**                       | Humble Hawksbill    | Installed via `rosdep` or apt; compiled against Python 3.11【860636887936188†L436-L460】. |
| **OpenVLA**                     | 1.0.x               | Custom install; commit hash should be recorded. |
| **Octo**                        | 0.x                 | Lightweight baseline; specify commit/branch. |
| **MoveIt 2**                    | Humble (0.x)        | Used for fallback planning. |
| **PyTorch**                     | 2.7.x               | Required for the policy runner. |
| **TensorFlow**                  | 2.15.x              | Optional; included for completeness. |
| **CUDA Toolkit**                | 12.x                | Should match driver. |
| **cuDNN**                       | 8.x                 | Included with CUDA toolkit. |
| **JupyterLab**                  | 4.x                 | For browser‑based orchestration. |
| **Docker**                      | 25.x                | Used for containerised services. |
| **docker‑compose**              | 2.x                 | Used to orchestrate multiple services. |

## Incompatibilities and Resolutions

During setup you may encounter version mismatches between the paper’s specification and the current Brev/Isaac Sim requirements.  All such conflicts should be documented here.  For example:

* The paper assumed **Python 3.8** and **ROS 2 Foxy**, but Isaac Sim 5.x requires **Python 3.11** and supports **ROS 2 Humble or Jazzy**【610528769204527†L425-L430】【860636887936188†L436-L460】.  We therefore upgraded the entire stack to Python 3.11 and installed ROS 2 Humble.
* The paper allowed an A100/H100 GPU, but Isaac Sim does not support GPUs without RT cores (A100/H100).  We used an RTX L40S instead.
* (Add any other discrepancies encountered during installation, with explanations of how they were resolved.)

## How to update

Whenever you upgrade or reinstall any component, revisit this file and update the version numbers accordingly.  This transparency ensures that anyone reproducing the results knows exactly what was installed and can replicate the environment precisely.