"""
Policy Package
--------------

Provides wrappers around vision–language–action (VLA) policies such as
OpenVLA and Octo.  These wrappers expose a common interface for loading
the model, conditioning it on observations and instructions, sampling
actions with uncertainty estimation (e.g. via Monte Carlo dropout or
ensemble sampling) and returning confidence scores.

The actual model implementations reside in separate repositories (e.g.
`openvla` and `octo`).  Here we assume those packages are installed and we
wrap them for use in the experiment pipeline.  When the real models are
unavailable, these classes fall back to dummy behaviour to enable end‑to‑end
testing without requiring GPU resources.
"""

from .base_policy import BasePolicy
from .openvla_policy import OpenVLA
from .octo_policy import Octo

__all__ = ["BasePolicy", "OpenVLA", "Octo"]