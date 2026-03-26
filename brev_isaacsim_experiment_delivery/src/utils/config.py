"""
Configuration Loading Utilities
------------------------------

This module provides helper functions for reading YAML configuration files and
validating their structure.  Config files define experiment parameters such as
task names, number of demonstrations, shift severities, model settings and
logging options.  The utilities here standardise the way configurations are
loaded across the experiment scripts.

Example:

```python
from src.utils.config import load_config

cfg = load_config('configs/pilot_config.yaml')
print(cfg['tasks']['drawer']['demo_count'])  # -> 10
```

If a required field is missing, `load_config` will raise a `KeyError`.  You can
extend the `validate_config` function to implement additional consistency
checks.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        A nested dictionary containing configuration parameters.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the file cannot be parsed as YAML.
    KeyError
        If required fields are missing (see `validate_config`).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate_config(data)
    return data


def validate_config(data: Dict[str, Any]) -> None:
    """Basic validation to ensure required top‑level keys exist.

    This function can be expanded to check for specific types,
    value ranges or internal consistency.  Currently it simply checks
    that the expected top‑level sections are present.

    Parameters
    ----------
    data : dict
        Loaded YAML configuration.

    Raises
    ------
    KeyError
        If a required section is missing.
    """
    required_keys = ["tasks", "policies", "gating", "logging"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required configuration section: {key}")