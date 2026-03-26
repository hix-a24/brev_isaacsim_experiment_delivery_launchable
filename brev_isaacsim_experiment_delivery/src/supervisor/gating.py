"""
Safety Gating State Machine
---------------------------

Implements the hysteresis state machine described in the paper【725427445771147†L470-L522】.
At each time step, the supervisor uses the calibrated risk estimate r_t to
determine whether to proceed with the learned policy, pause and re‑observe,
or fall back to the planner.  The state machine has three states:

1. **Proceed** – The policy is confident (r_t < δ_low) and actions are executed.
2. **Pause/Reobserve** – Moderate risk (δ_low ≤ r_t < δ_high); the system pauses,
   takes a new observation and recomputes r_t.  If risk drops below δ_low
   the state transitions back to Proceed; if it rises above δ_high it
   transitions to Fallback.
3. **Fallback** – High risk (r_t ≥ δ_high); control is handed over to the
   fallback planner until the planner finishes or decides to abort.

The thresholds δ_low and δ_high are specified in the configuration (e.g.
δ_low=0.2 and δ_high=0.5).  The state machine also returns events that
should be logged (e.g. transitions to pause or fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class ControlState(Enum):
    """Possible control states for the gating state machine."""
    PROCEED = auto()
    PAUSE_REOBSERVE = auto()
    FALLBACK = auto()


@dataclass
class SafetyGatingStateMachine:
    """Hysteresis state machine controlling policy execution based on risk."""

    delta_low: float = 0.2
    delta_high: float = 0.5
    current_state: ControlState = ControlState.PROCEED

    def update(self, risk: float) -> Tuple[ControlState, Optional[str]]:
        """Update the state based on the new risk value.

        Parameters
        ----------
        risk : float
            Calibrated risk estimate r_t ∈ [0, 1].

        Returns
        -------
        state : ControlState
            New control state.
        event : Optional[str]
            Event name to log, if a transition occurred ("pause", "fallback",
            "resume").  None if the state is unchanged.
        """
        prev_state = self.current_state
        event: Optional[str] = None

        if self.current_state == ControlState.PROCEED:
            if risk >= self.delta_high:
                self.current_state = ControlState.FALLBACK
                event = "fallback"
            elif risk >= self.delta_low:
                self.current_state = ControlState.PAUSE_REOBSERVE
                event = "pause"
        elif self.current_state == ControlState.PAUSE_REOBSERVE:
            # After a re‑observe, update state based on risk again
            if risk < self.delta_low:
                self.current_state = ControlState.PROCEED
                event = "resume"
            elif risk >= self.delta_high:
                self.current_state = ControlState.FALLBACK
                event = "fallback"
        elif self.current_state == ControlState.FALLBACK:
            # Remain in fallback until the planner signals completion
            if risk < self.delta_low:
                self.current_state = ControlState.PROCEED
                event = "resume"
        # else: unknown state

        return self.current_state, event