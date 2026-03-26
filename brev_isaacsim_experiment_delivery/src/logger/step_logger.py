"""
Step Logger
-----------

Records per‑time‑step data during an episode, including the robot state,
action, raw and calibrated confidence values, risk estimate and gating
state.  The step log is used to generate calibration curves and analyze
intervention timing.  Each episode’s steps can be stored in a separate
file if desired, or concatenated into one large CSV/Parquet file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StepLogger:
    """Logger for step‑level metrics."""

    output_dir: Path
    filename: str = "steps.csv"
    fieldnames: List[str] = field(default_factory=lambda: [
        "episode_id", "t_step", "phase", "ee_pose", "joint_positions",
        "raw_policy_confidence", "raw_uncertainty", "uncertainty_entropy",
        "uncertainty_variance", "calibrated_success_prob", "calibrated_failure_risk",
        "gating_state", "action_vector", "observation_frame_path",
    ])
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.filename
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_step(self, data: Dict[str, Any]) -> None:
        missing = set(self.fieldnames) - set(data.keys())
        if missing:
            raise KeyError(f"Missing step fields: {missing}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path)