"""
Contact Logger
--------------

Logs contact events between the robot and the environment.  Each contact
record includes the episode and time step, the robot link involved, the
object or environment component contacted, the contact force and torque, the
incident type (e.g. gentle touch vs. collision) and the signed distance
margin to collision.  These logs enable the analysis of collision rates,
force distributions and safety margins for Figure 10.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ContactLogger:
    """Logger for contact and proximity events."""

    output_dir: Path
    filename: str = "contacts.csv"
    fieldnames: List[str] = field(default_factory=lambda: [
        "episode_id", "t_step", "link_name", "other_object", "contact_force_n",
        "contact_torque_nm", "incident_type", "margin_to_collision_m",
    ])
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.filename
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_contact(self, data: Dict[str, Any]) -> None:
        missing = set(self.fieldnames) - set(data.keys())
        if missing:
            raise KeyError(f"Missing contact fields: {missing}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path)