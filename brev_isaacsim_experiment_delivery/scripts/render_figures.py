#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.render_figures import render_all_figures


if __name__ == "__main__":
    render_all_figures(ROOT / "data_logs", ROOT / "figures")
    print(f"Rendered figures into {ROOT / 'figures'}")
