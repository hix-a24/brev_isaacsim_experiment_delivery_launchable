"""
Calibrator Module
-----------------

Provides a simple temperature scaling implementation for calibrating the
policy’s raw success probability estimates.  Temperature scaling fits a
single scalar parameter τ such that the softmax (or sigmoid) outputs of a
model better reflect true probabilities.  During experiments, a calibration
dataset is collected (e.g. 200 episodes) and the calibrator is fitted on
the mapping from raw model confidence to binary success outcomes.

References:
  - Guo et al., “On Calibration of Modern Neural Networks” (2017)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class TemperatureCalibrator:
    """Fits a temperature scaling model to map raw scores to calibrated probabilities."""

    temperature: float = 1.0
    fitted: bool = False

    def fit(self, logits: Iterable[float], outcomes: Iterable[int]) -> float:
        """Fit the temperature parameter on a set of logits and binary outcomes.

        Parameters
        ----------
        logits : iterable of float
            Raw logit or confidence values from the policy (before sigmoid).
        outcomes : iterable of int
            Binary success outcomes (1 for success, 0 for failure).

        Returns
        -------
        float
            The fitted temperature parameter.
        """
        logits = np.array(list(logits), dtype=np.float64)
        labels = np.array(list(outcomes), dtype=np.float64)

        # Use negative log likelihood as objective
        def nll(temp: float) -> float:
            temp = max(temp, 1e-6)
            scaled = np.clip(-logits / temp, -50.0, 50.0)
            prob = 1 / (1 + np.exp(scaled))
            # Avoid log(0)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            return -np.mean(labels * np.log(prob) + (1 - labels) * np.log(1 - prob))

        res = minimize(lambda t: nll(t[0]), x0=[1.0], bounds=[(1e-3, 100)])
        self.temperature = res.x[0]
        self.fitted = True
        return self.temperature

    def transform(self, logits: Iterable[float]) -> List[float]:
        """Apply temperature scaling to raw logits.

        Returns calibrated success probabilities.
        """
        if not self.fitted:
            raise RuntimeError("TemperatureCalibrator must be fitted before use.")
        logits = np.array(list(logits), dtype=np.float64)
        scaled = np.clip(-logits / self.temperature, -50.0, 50.0)
        prob = 1 / (1 + np.exp(scaled))
        return prob.tolist()