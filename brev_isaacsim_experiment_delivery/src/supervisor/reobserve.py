"""
Reobserve Strategy
------------------

During the pause/reobserve state, the system stops executing actions and
captures a new observation from the environment to reduce uncertainty.  This
module defines a `ReobserveStrategy` class that can be called by the
supervisor to implement this behaviour.  The default strategy simply waits
one simulation time step and returns the current observation; more advanced
strategies could move the camera or end effector to a better viewpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any


@dataclass
class ReobserveStrategy:
    """A strategy for obtaining a new observation when risk is moderate."""

    get_observation: Callable[[], Dict[str, Any]]

    def __call__(self) -> Dict[str, Any]:
        """Return a fresh observation to re‑evaluate risk.

        In practice, this may involve waiting for the next camera frame,
        moving the robot to improve visibility, or averaging multiple frames.
        Here we simply call `get_observation` directly.
        """
        return self.get_observation()