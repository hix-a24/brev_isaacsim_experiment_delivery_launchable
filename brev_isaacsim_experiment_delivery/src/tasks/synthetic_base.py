from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PHASE_THRESHOLDS = [0.25, 0.5, 0.75, 1.0]
SHIFT_KEYS = ["lighting", "texture", "occlusion", "sensor", "combined"]


@dataclass
class SyntheticTaskBase:
    """Synthetic manipulation task used when Isaac Sim is unavailable.

    The class exposes a task-like interface compatible with the existing
    project, while simulating progress, contact events, and shift-dependent
    difficulty using a lightweight numerical model.
    """

    config: Dict[str, Any]
    seed: Optional[int] = None
    task_name: str = "generic"
    phases: List[str] = field(default_factory=lambda: ["approach", "manipulate", "place", "finish"])
    base_action_templates: np.ndarray = field(default_factory=lambda: np.eye(4, 7, dtype=np.float32))
    success_force_limit: float = 9.5
    max_collisions: int = 3
    max_steps: int = 40

    rng: np.random.Generator = field(init=False)
    shift_axis: str = field(init=False, default="lighting")
    shift_severity: int = field(init=False, default=0)
    shift_factor: float = field(init=False, default=0.0)
    progress: float = field(init=False, default=0.0)
    collisions: int = field(init=False, default=0)
    step_count: int = field(init=False, default=0)
    success: bool = field(init=False, default=False)
    terminated: bool = field(init=False, default=False)
    failure_reason: str = field(init=False, default="")
    last_info: Dict[str, Any] = field(init=False, default_factory=dict)
    last_action: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.max_steps = int(self.config.get("max_steps", self.max_steps))
        self.last_action = np.zeros(7, dtype=np.float32)
        self._init_simulator()

    def _init_simulator(self) -> None:
        self.world = {"backend": "synthetic", "task": self.task_name}

    @property
    def phase_index(self) -> int:
        idx = min(int(self.progress * len(self.phases)), len(self.phases) - 1)
        return max(idx, 0)

    @property
    def phase_name(self) -> str:
        return self.phases[self.phase_index]

    @property
    def current_goal_name(self) -> str:
        return self.phase_name

    def _resolve_shift(self, shift_config: Dict[str, Any]) -> Tuple[str, int, float]:
        axis = str(shift_config.get("axis", "lighting"))
        if axis not in SHIFT_KEYS:
            axis = "combined"
        severity = int(shift_config.get("severity", 0))
        severity = int(np.clip(severity, 0, 3))
        axis_bonus = {
            "lighting": 0.05,
            "texture": 0.07,
            "occlusion": 0.11,
            "sensor": 0.09,
            "combined": 0.16,
        }[axis]
        shift_factor = float(np.clip(severity * 0.12 + axis_bonus, 0.0, 0.65))
        return axis, severity, shift_factor

    def reset(self, shift_config: Dict[str, Any]) -> Dict[str, Any]:
        self.shift_axis, self.shift_severity, self.shift_factor = self._resolve_shift(shift_config)
        self.progress = 0.0
        self.collisions = 0
        self.step_count = 0
        self.success = False
        self.terminated = False
        self.failure_reason = ""
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_info = {
            "phase": self.phase_name,
            "collision": False,
            "contact_force_n": 0.0,
            "contact_torque_nm": 0.0,
            "incident_type": "none",
            "margin_to_collision_m": 0.08,
            "world_x": 0.0,
            "world_y": 0.0,
            "world_z": 0.2,
            "progress": self.progress,
        }
        return self._build_observation()

    def _build_observation(self) -> Dict[str, Any]:
        phase_norm = self.phase_index / max(len(self.phases) - 1, 1)
        severity_norm = self.shift_severity / 3.0
        collision_norm = min(self.collisions / max(self.max_collisions, 1), 1.0)
        safety_margin = float(np.clip(0.09 - 0.03 * severity_norm - 0.015 * collision_norm - 0.01 * self.shift_factor, 0.005, 0.12))
        time_remaining = 1.0 - min(self.step_count / max(self.max_steps, 1), 1.0)
        task_bias = self.task_bias()
        state = np.array(
            [
                float(np.clip(self.progress, 0.0, 1.0)),
                float(phase_norm),
                float(severity_norm),
                float(self.shift_factor),
                float(collision_norm),
                float(safety_margin),
                float(time_remaining + task_bias),
            ],
            dtype=np.float32,
        )
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = np.uint8(np.clip(255 * state[0], 0, 255))
        rgb[..., 1] = np.uint8(np.clip(255 * state[2], 0, 255))
        rgb[..., 2] = np.uint8(np.clip(255 * state[5] / 0.12, 0, 255))
        depth = np.full((32, 32), fill_value=max(0.05, 1.0 - state[0]), dtype=np.float32)
        return {
            "rgb": rgb,
            "depth": depth,
            "state": state,
            "task": self.task_name,
            "phase": self.phase_name,
        }

    def task_bias(self) -> float:
        return 0.0

    def oracle_action(self, observation: Dict[str, Any]) -> np.ndarray:
        template = self.base_action_templates[self.phase_index].copy()
        state = np.asarray(observation["state"], dtype=np.float32)
        modifier = np.array(
            [
                0.10 * (state[0] - 0.5),
                0.08 * (0.5 - state[2]),
                0.06 * (state[5] - 0.04),
                -0.04 * state[4],
                0.03 * state[6],
                -0.02 * state[3],
                0.01,
            ],
            dtype=np.float32,
        )
        action = np.tanh(template + modifier)
        return action.astype(np.float32)

    def oracle_confidence(self, observation: Dict[str, Any]) -> float:
        state = np.asarray(observation["state"], dtype=np.float32)
        confidence = 0.88 + 0.10 * state[0] + 0.18 * state[5] - 0.35 * state[2] - 0.25 * state[3] - 0.10 * state[4]
        return float(np.clip(confidence, 0.05, 0.98))

    def _action_quality(self, action: np.ndarray, observation: Dict[str, Any]) -> float:
        oracle = self.oracle_action(observation)
        action = np.asarray(action, dtype=np.float32)
        denom = float(np.linalg.norm(oracle) * np.linalg.norm(action) + 1e-6)
        cosine = float(np.dot(oracle, action) / denom)
        l1_error = float(np.mean(np.abs(oracle - action)))
        quality = 0.65 * ((cosine + 1.0) / 2.0) + 0.35 * max(0.0, 1.0 - l1_error / 1.2)
        return float(np.clip(quality, 0.0, 1.0))

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.terminated:
            return self._build_observation(), 0.0, True, dict(self.last_info)

        observation = self._build_observation()
        action = np.asarray(action, dtype=np.float32)
        quality = self._action_quality(action, observation)
        severity_norm = self.shift_severity / 3.0
        progress_delta = float(np.clip(0.03 + 0.14 * quality - 0.05 * severity_norm - 0.03 * self.shift_factor + self.task_progress_bonus(), -0.02, 0.18))
        progress_delta += float(self.rng.normal(0.0, 0.006))
        self.progress = float(np.clip(self.progress + progress_delta, 0.0, 1.05))
        self.step_count += 1

        collision_prob = float(np.clip(0.01 + 0.18 * (1.0 - quality) + 0.08 * severity_norm + 0.05 * self.shift_factor, 0.0, 0.75))
        gentle_contact = self.rng.random() < collision_prob
        is_collision = gentle_contact and (self.rng.random() < (0.25 + 0.45 * severity_norm))
        if is_collision:
            self.collisions += 1

        contact_force = 0.0
        contact_torque = 0.0
        incident_type = "none"
        if gentle_contact:
            base_force = 1.5 + 2.0 * severity_norm + 3.0 * (1.0 - quality) + 1.2 * is_collision
            contact_force = float(max(abs(self.rng.normal(base_force, 0.6)), 0.05))
            contact_torque = float(max(abs(self.rng.normal(0.18 * base_force, 0.12)), 0.01))
            incident_type = "collision" if is_collision else "gentle_touch"

        margin = float(np.clip(0.09 - 0.05 * collision_prob - 0.02 * severity_norm + self.rng.normal(0.0, 0.004), 0.0, 0.12))
        reward = progress_delta - (0.15 if is_collision else 0.0)

        if self.progress >= 1.0 and self.collisions <= self.max_collisions and contact_force < self.success_force_limit:
            self.success = True
            self.terminated = True
        elif self.collisions > self.max_collisions:
            self.failure_reason = "collision"
            self.terminated = True
        elif self.step_count >= self.max_steps:
            self.failure_reason = "timeout"
            self.terminated = True
        elif contact_force > self.success_force_limit:
            self.failure_reason = "unsafe_force"
            self.terminated = True

        self.last_action = action
        self.last_info = {
            "phase": self.phase_name,
            "collision": bool(is_collision),
            "contact_force_n": round(contact_force, 6),
            "contact_torque_nm": round(contact_torque, 6),
            "incident_type": incident_type,
            "margin_to_collision_m": round(margin, 6),
            "world_x": round(float(self.rng.uniform(-0.5, 0.5) + 0.2 * (self.progress - 0.5)), 6),
            "world_y": round(float(self.rng.uniform(-0.45, 0.45) + 0.08 * (self.shift_severity - 1)), 6),
            "world_z": round(float(0.15 + 0.25 * (1.0 - self.progress) + self.rng.uniform(-0.03, 0.03)), 6),
            "progress": round(self.progress, 6),
            "quality": round(quality, 6),
            "task_success": self.success,
            "task_failed": self.terminated and not self.success,
            "failure_reason": self.failure_reason,
        }
        return self._build_observation(), float(reward), bool(self.terminated), dict(self.last_info)

    def task_progress_bonus(self) -> float:
        return 0.0

    def check_success(self) -> bool:
        return bool(self.success)

    def close(self) -> None:
        self.world = None
