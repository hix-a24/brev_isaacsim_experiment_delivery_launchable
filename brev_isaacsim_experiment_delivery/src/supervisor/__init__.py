"""
Supervisor Package
------------------

This package implements the safety supervision system that sits on top of
the VLA policy.  It contains routines for calibrating the policy’s raw
confidence outputs into risk probabilities, estimating failure risk, and
deciding whether to proceed with the learned policy, pause for a re‑observation
or trigger a fallback planner.  The main components are:

* `calibrator` – Fits a temperature scaling model to map raw confidence to
  calibrated success probabilities using validation data.
* `risk_estimator` – Computes a scalar risk metric `r_t` from the calibrated
  success probability and optional uncertainty measures.
* `gating` – Implements a hysteresis state machine with thresholds δ_low and
  δ_high for deciding the control state at each time step.
* `reobserve` – Implements the pause/reobserve behaviour to obtain a new
  observation when risk is moderate.
"""

from .calibrator import TemperatureCalibrator
from .risk_estimator import RiskEstimator
from .gating import SafetyGatingStateMachine
from .reobserve import ReobserveStrategy

__all__ = [
    "TemperatureCalibrator",
    "RiskEstimator",
    "SafetyGatingStateMachine",
    "ReobserveStrategy",
]