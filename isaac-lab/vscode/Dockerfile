# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/isaac-lab:2.3.0

RUN apt update &&     apt install -y       build-essential       cmake       curl       dnsutils       git       iproute2       jq       less       python3-pip       tmux       tree       unzip &&     rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.96.4

RUN code-server     --install-extension ms-python.python     --install-extension ms-toolsai.jupyter     --force /tmp

COPY entrypoint.sh /entrypoint.sh

RUN mkdir -p /workspace /home/isaac-sim/.cache/ov /home/isaac-sim/.local/share/ov &&     chown -R isaac-sim:isaac-sim /workspace /home/isaac-sim/.cache/ov /home/isaac-sim/.local/share/ov /isaac-sim/kit/cache

RUN cp /root/.bashrc /isaac-sim/.root-bashrc &&     echo 'source $HOME/.root-bashrc' >> /isaac-sim/.bashrc

USER isaac-sim:isaac-sim

ENTRYPOINT ["/bin/bash"]
CMD ["/entrypoint.sh"]
