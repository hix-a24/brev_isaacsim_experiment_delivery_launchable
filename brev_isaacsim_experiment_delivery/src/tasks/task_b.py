"""Synthetic clutter-sort task compatible with the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .synthetic_base import SyntheticTaskBase


@dataclass
class ClutterSortTask(SyntheticTaskBase):
    config: Dict[str, Any]
    seed: Optional[int] = None
    task_name: str = "clutter_sort"
    phases: List[str] = field(default_factory=lambda: ["scan_scene", "grasp_target", "route_to_tray", "sort_finalize"])
    base_action_templates: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.25, 0.75, 0.10, -0.20, 0.05, 0.20, -0.05],
                [0.60, 0.35, -0.15, 0.30, -0.10, 0.30, 0.55],
                [-0.20, 0.10, 0.85, -0.15, 0.25, 0.45, -0.20],
                [-0.45, -0.35, -0.10, 0.20, 0.55, -0.10, 0.15],
            ],
            dtype=np.float32,
        )
    )
    success_force_limit: float = 8.8
    max_collisions: int = 2
    max_steps: int = 42

    def task_bias(self) -> float:
        return -0.02

    def task_progress_bonus(self) -> float:
        if self.phase_name == "scan_scene":
            return -0.005
        return 0.005
