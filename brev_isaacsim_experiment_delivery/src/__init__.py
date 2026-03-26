"""
src package
=================

This package contains the implementation of the uncertainty-calibrated safety
gating experiment scaffold used by the Brev launchable.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version(__package__ or __name__)
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"
