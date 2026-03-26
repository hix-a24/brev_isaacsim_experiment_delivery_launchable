"""Synthetic drawer-object task compatible with the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .synthetic_base import SyntheticTaskBase


@dataclass
class DrawerObjectTask(SyntheticTaskBase):
    config: Dict[str, Any]
    seed: Optional[int] = None
    task_name: str = "drawer"
    phases: List[str] = field(default_factory=lambda: ["open_drawer", "grasp_block", "place_in_bin", "close_drawer"])
    base_action_templates: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.85, -0.25, 0.10, 0.20, 0.05, 0.00, 0.15],
                [0.35, 0.65, -0.10, 0.25, -0.05, 0.15, 0.55],
                [-0.45, 0.25, 0.72, -0.20, 0.10, 0.35, -0.30],
                [-0.70, -0.15, -0.25, 0.10, 0.40, -0.20, 0.05],
            ],
            dtype=np.float32,
        )
    )
    success_force_limit: float = 10.0
    max_collisions: int = 3
    max_steps: int = 36

    def task_bias(self) -> float:
        return 0.03

    def task_progress_bonus(self) -> float:
        if self.phase_name == "close_drawer":
            return 0.01
        return 0.0
