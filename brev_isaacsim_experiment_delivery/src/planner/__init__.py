"""
Planner Package
---------------

Provides a high‑level interface to the fallback motion planner.  The
planner’s role is to compute safe trajectories and execute them when the
gating logic indicates that the learned policy is too risky.  In our
implementation, we rely on MoveIt 2 with OMPL for planning.  However, the
`fallback_planner.py` module contains a simplified placeholder that can be
replaced with a proper MoveIt 2 integration.
"""

from .fallback_planner import FallbackPlanner

__all__ = ["FallbackPlanner"]