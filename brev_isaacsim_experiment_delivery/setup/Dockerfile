FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# This Dockerfile provides a base image for running the simulation, policy and
# supervisor services.  It installs a minimal set of packages, copies the
# repository contents and sets up a Python environment.  The actual policy
# training and simulation require additional dependencies installed via
# `pip install -r requirements.txt` or similar; see `setup/install.sh` for the full
# host installation instructions.

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      curl git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install numpy scipy pandas matplotlib ruamel.yaml scikit-learn && \
    /app/venv/bin/pip install jupyterlab && \
    echo "[Dockerfile] Base dependencies installed.  Additional packages should be installed in runtime scripts."

ENV PATH="/app/venv/bin:$PATH"

CMD ["bash"]