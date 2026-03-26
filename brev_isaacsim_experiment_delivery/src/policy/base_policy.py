"""Base utilities for lightweight synthetic VLA policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


class BasePolicy(ABC):
    """Abstract interface for VLA-style policies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.model: Dict[str, Any] | None = None
        self.instruction: str = ""
        self.seed = int(self.config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.asarray(x)))

    def _resolve_checkpoint_path(self) -> Path:
        checkpoint = self.config.get("checkpoint", "")
        path = Path(checkpoint)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    def _default_model(self) -> Dict[str, Any]:
        return {
            "action_weights": np.eye(7, dtype=np.float32),
            "action_bias": np.zeros(7, dtype=np.float32),
            "confidence_weights": np.array([1.2, -0.5, -1.0, -0.8, -0.4, 1.0, 0.2], dtype=np.float32),
            "confidence_bias": np.float32(0.2),
            "noise_scale": np.float32(0.05),
            "calibration_temperature": np.float32(1.0),
        }

    def calibration_temperature(self) -> float:
        if self.model is None:
            return 1.0
        return float(self.model.get("calibration_temperature", 1.0))

    def _state_vector(self, observation: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation.get("state", np.zeros(7, dtype=np.float32)), dtype=np.float32)
        if state.shape[0] != 7:
            padded = np.zeros(7, dtype=np.float32)
            padded[: min(7, state.shape[0])] = state[:7]
            state = padded
        return state

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self, instruction: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError
