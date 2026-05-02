"""Feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered numeric features used for modeling."""
    df = df.copy()

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(
        df.columns
    ):
        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    if {"OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}.issubset(
        df.columns
    ):
        df["TotalPorchSF"] = (
            df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        )

    if {"WoodDeckSF", "PoolArea"}.issubset(df.columns):
        df["OutdoorSF"] = df["WoodDeckSF"] + df.get("TotalPorchSF", 0) + df[
            "PoolArea"
        ]

    if {"GrLivArea", "TotRmsAbvGrd"}.issubset(df.columns):
        df["RoomDensity"] = df["GrLivArea"] / df["TotRmsAbvGrd"].replace(0, np.nan)

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["Age"] = df["YrSold"] - df["YearBuilt"]

    return df


def drop_highly_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pre-identified highly correlated features."""
    df = df.copy()
    to_drop = ["YearBuilt", "GarageArea", "GrLivArea"]
    return df.drop(columns=[col for col in to_drop if col in df.columns])
