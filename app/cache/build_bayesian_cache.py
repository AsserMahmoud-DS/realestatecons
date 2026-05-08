"""Build fallback feature-value cache for Bayesian Ridge app inference.

The cache is computed from the full available training dataset after applying the
same pre-model dataflow used in the probabilistic notebook:
1) cleaning
2) feature engineering
3) correlated-feature drop
4) IQR outlier row removal (<4% outlier-rate columns)

It saves median/mode defaults that can fill sparse user/chatbot inputs.
"""



import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys

BASE_DIR = Path('/home/amraas/projects/realestatecons')

sys.path.append(str(BASE_DIR))

from src.data.preprocess import clean_train_data
from src.features.features import (
    add_engineered_features,
    drop_highly_correlated_features,
    drop_iqr_outliers_for_low_rate_columns,
)

TARGET_COL = "SalePrice"


def _first_mode(series: pd.Series) -> Any:
    mode = series.mode(dropna=True)
    if mode.empty:
        return None
    value = mode.iloc[0]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return value


def _compute_defaults(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    defaults: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            median = series.median()
            defaults[col] = {
                "strategy": "median",
                "value": _to_python_scalar(median) if pd.notna(median) else None,
                "dtype": str(series.dtype),
            }
        else:
            defaults[col] = {
                "strategy": "mode",
                "value": _to_python_scalar(_first_mode(series)),
                "dtype": str(series.dtype),
            }
    return defaults


def build_cache_payload(train_csv_path: Path) -> dict[str, Any]:
    train_df_raw = pd.read_csv(train_csv_path)

    cleaned = clean_train_data(train_df_raw)
    raw_feature_frame = cleaned.drop(columns=[TARGET_COL]).copy()

    engineered = add_engineered_features(cleaned)
    engineered = drop_highly_correlated_features(engineered)

    modeling_df = drop_iqr_outliers_for_low_rate_columns(
        engineered,
        target_col=TARGET_COL,
        outlier_rate_threshold_pct=4.0,
    )
    model_input_frame = modeling_df.drop(columns=[TARGET_COL]).copy()

    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_col": TARGET_COL,
            "outlier_rate_threshold_pct": 4.0,
            "dataflow": [
                "clean_train_data",
                "add_engineered_features",
                "drop_highly_correlated_features",
                "drop_iqr_outliers_for_low_rate_columns",
            ],
            "row_counts": {
                "raw_train_rows": int(len(train_df_raw)),
                "post_clean_rows": int(len(cleaned)),
                "post_outlier_rows": int(len(modeling_df)),
            },
            "column_counts": {
                "raw_feature_columns": int(raw_feature_frame.shape[1]),
                "model_input_columns": int(model_input_frame.shape[1]),
            },
        },
        "raw_input_defaults": _compute_defaults(raw_feature_frame),
        "model_input_defaults": _compute_defaults(model_input_frame),
    }
    return payload


def write_cache(payload: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    train_csv_path = base_dir / "data" / "raw" / "train.csv"
    cache_path = base_dir / "app" / "cache" / "bayesian_defaults.json"

    payload = build_cache_payload(train_csv_path)
    write_cache(payload, cache_path)
    print(f"Saved Bayesian cache to: {cache_path}")


if __name__ == "__main__":
    main()
