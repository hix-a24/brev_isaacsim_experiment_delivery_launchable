#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config",
        "configs/pilot_config.yaml",
        "--clean",
        "--episodes-per-condition",
        "2",
        "--max-steps",
        "20",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))
