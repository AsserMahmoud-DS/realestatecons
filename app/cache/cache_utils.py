"""Utilities for loading and applying app-side fallback cache values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_bayesian_defaults(cache_path: Path) -> dict[str, Any]:
    """Load cached Bayesian fallback values from JSON."""
    return json.loads(cache_path.read_text())


def apply_defaults(values: dict[str, Any], defaults_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fill missing/empty user fields from cached defaults."""
    merged = dict(values)
    for key, spec in defaults_map.items():
        if key not in merged or merged[key] in (None, ""):
            merged[key] = spec.get("value")
    return merged


def to_single_row_frame(values: dict[str, Any]) -> pd.DataFrame:
    """Convert an input dictionary into a single-row dataframe for prediction."""
    return pd.DataFrame([values])
