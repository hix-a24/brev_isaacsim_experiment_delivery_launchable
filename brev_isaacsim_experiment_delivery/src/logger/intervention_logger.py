"""
Intervention Logger
-------------------

Logs events where the safety supervisor intervenes: pause/reobserve, fallback,
or safe stop.  For each event we record the episode and time step, the type
of intervention, the risk estimate before and after the event, the robot’s
world coordinates at the time of intervention, and whether the intervention
resolved the risk.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class InterventionLogger:
    """Logger for intervention events."""

    output_dir: Path
    filename: str = "interventions.csv"
    fieldnames: List[str] = field(default_factory=lambda: [
        "episode_id", "t_step", "event_type", "phase", "risk_before", "risk_after",
        "world_x", "world_y", "world_z", "resolved_by", "resume_success",
    ])
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.filename
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_intervention(self, data: Dict[str, Any]) -> None:
        missing = set(self.fieldnames) - set(data.keys())
        if missing:
            raise KeyError(f"Missing intervention fields: {missing}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path)