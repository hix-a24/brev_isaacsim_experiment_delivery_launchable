"""Lightweight fallback planner for the runnable synthetic stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class FallbackPlanner:
    config: Dict[str, Any]

    def plan_and_execute(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        oracle_action = np.asarray(context.get("oracle_action", np.zeros(7)), dtype=np.float32)
        risk = float(context.get("risk", 0.5))
        margin = float(context.get("margin_to_collision_m", 0.04))
        phase = str(context.get("phase", goal))

        retreat = np.array([-0.25, -0.15, 0.20, 0.05, 0.10, -0.05, 0.0], dtype=np.float32)
        blend = float(np.clip(0.35 + 0.6 * risk, 0.35, 0.95))
        action = np.tanh((1.0 - blend) * oracle_action + blend * retreat).astype(np.float32)
        if phase in {"place_in_bin", "route_to_tray", "sort_finalize", "close_drawer"}:
            action = np.tanh(0.75 * oracle_action + 0.25 * action).astype(np.float32)

        path_length = float(0.35 + 0.8 * np.linalg.norm(action))
        planning_time = float(0.08 + 0.65 * risk + 0.2 * max(0.03 - margin, 0.0))
        risk_after = float(np.clip(risk - 0.28 - 0.4 * margin, 0.02, 0.85))
        success = risk_after < 0.65

        return {
            "status": "success" if success else "partial",
            "time": round(planning_time, 6),
            "path_length": round(path_length, 6),
            "action": action,
            "risk_after": round(risk_after, 6),
            "goal": goal,
        }
