"""
Risk Estimator
--------------

Defines how to map calibrated success probabilities and uncertainty metrics
to a scalar risk value r_t used by the gating logic.  A simple formulation
is:

    r_t = 1 - p_success

where p_success is the calibrated probability that the policy succeeds in the
current state.  Additional factors such as uncertainty variance or entropy
may also contribute to risk; for example:

    r_t = (1 - p_success) + alpha * variance + beta * entropy

where `alpha` and `beta` are configurable weights.  The parameters can be
set via the YAML config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RiskEstimator:
    """Compute a scalar risk metric from calibrated success probability and uncertainty."""

    alpha: float = 1.0  # weight for variance
    beta: float = 1.0   # weight for entropy

    def estimate(self, calibrated_p_success: float, info: Dict[str, float]) -> float:
        """Compute r_t given calibrated success probability and uncertainty metrics.

        Parameters
        ----------
        calibrated_p_success : float
            The calibrated probability that the policy will succeed.
        info : dict
            Additional metrics from the policy, including at least
            `uncertainty_variance` and `uncertainty_entropy`.

        Returns
        -------
        float
            A scalar risk between 0 and 1, where 1 indicates highest risk.
        """
        variance = float(info.get("uncertainty_variance", 0.0))
        entropy = float(info.get("uncertainty_entropy", 0.0))
        # Compute risk as one minus success probability plus weighted uncertainty
        risk = (1.0 - calibrated_p_success) + self.alpha * variance + self.beta * entropy
        # Clamp to [0, 1]
        return float(np.clip(risk, 0.0, 1.0))