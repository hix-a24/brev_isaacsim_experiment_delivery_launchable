"""
Episode Logger
--------------

Records high‑level summary information per episode.  Each call to
`log_episode` appends a new row to a CSV/Parquet file.  Fields include run
identifier, task name, method, success outcome, number of interventions,
collision statistics and other metrics described in the paper.  See
`data_logs/SCHEMA.md` for details.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class EpisodeLogger:
    """Logger for episode‑level metrics."""

    output_dir: Path
    filename: str = "episodes.csv"
    fieldnames: List[str] = field(default_factory=lambda: [
        "run_id", "task", "method", "policy", "seed", "shift_axis", "shift_severity", "instruction",
        "success", "failure_reason", "collision_any", "collision_count", "peak_contact_force_n",
        "peak_contact_torque_nm", "num_interventions", "num_pause_reobserve", "num_fallbacks",
        "episode_wall_time_s", "sim_time_s", "object_in_bin", "drawer_closed", "timed_out",
        "video_path",
    ])
    log_path: Path = field(init=False)
    _csv_file: Optional[Any] = field(init=False, default=None)
    _csv_writer: Optional[csv.DictWriter] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.filename
        # If the file does not exist, create it and write the header
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_episode(self, data: Dict[str, Any]) -> None:
        """Append a row to the episode log.

        Parameters
        ----------
        data : dict
            A dictionary containing all required fields defined in `fieldnames`.
        """
        missing = set(self.fieldnames) - set(data.keys())
        if missing:
            raise KeyError(f"Missing episode fields: {missing}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def to_dataframe(self) -> pd.DataFrame:
        """Load the CSV into a pandas DataFrame."""
        return pd.read_csv(self.log_path)