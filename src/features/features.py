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


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal and nominal categorical features into numeric columns.

    Ordinal features are mapped to ordered integer codes. Nominal features are
    one-hot encoded. Missing columns are skipped.
    """
    df = df.copy()

    ordinal_categories = {
        "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtCond": ["Po", "Fa", "TA", "Gd", "Ex"],
        "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
        "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        "FireplaceQu": ["Po", "Fa", "TA", "Gd", "Ex"],
        "GarageQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        "GarageCond": ["Po", "Fa", "TA", "Gd", "Ex"],
        "PoolQC": ["Fa", "TA", "Gd", "Ex"],
        "BsmtExposure": ["No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        "GarageFinish": ["Unf", "RFn", "Fin"],
        "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
        "LotShape": ["IR3", "IR2", "IR1", "Reg"],
        "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
        "LandSlope": ["Sev", "Mod", "Gtl"],
        "PavedDrive": ["N", "P", "Y"],
    }

    ordinal_cols = [col for col in ordinal_categories if col in df.columns]
    for col in ordinal_cols:
        categories = ordinal_categories[col]
        cat = pd.Categorical(df[col], categories=categories, ordered=True)
        # df[col] = cat.codes.replace(-1, np.nan)
        df[col] = np.where(cat.codes == -1, np.nan, cat.codes)
    categorical_cols = [
        col
        for col in df.columns
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])
    ]
    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    return df


def add_log_transformed_features(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Add log1p-transformed versions of selected numeric columns and drop originals.

    Skips missing or non-numeric columns. If a column has values <= -1, it is
    shifted up before log1p so the transform remains valid.
    """
    df = df.copy()
    transformed_originals: list[str] = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].astype(float)
            min_value = series.min()
            if pd.notna(min_value) and min_value <= -1.0:
                shift = abs(min_value) + 1.0
                series = series + shift
            df[f"{col}_log"] = np.log1p(series)
            transformed_originals.append(col)
    if transformed_originals:
        df = df.drop(columns=transformed_originals)
    return df


def drop_iqr_outliers_for_low_rate_columns(
    df: pd.DataFrame,
    *,
    target_col: str = "SalePrice",
    outlier_rate_threshold_pct: float = 10.0,
) -> pd.DataFrame:
    """Drop rows with IQR outliers from columns whose outlier rate is below threshold.

    This mirrors the rule used in the EDA features notebook:
    1. For each numeric feature (excluding ``target_col``), compute IQR outliers.
    2. Keep only columns where outlier percentage is strictly less than
       ``outlier_rate_threshold_pct``.
    3. Drop any row that is an outlier in at least one of those selected columns.
    """
    cleaned_df = df.copy()

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]

    outlier_summary: list[dict[str, float | str]] = []
    for col in numeric_cols:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (cleaned_df[col] < lower) | (cleaned_df[col] > upper)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            outlier_summary.append(
                {
                    "feature": col,
                    "outlier_pct": 100 * outlier_count / len(cleaned_df),
                }
            )

    if not outlier_summary:
        return cleaned_df

    outlier_summary_df = pd.DataFrame(outlier_summary)
    drop_candidate_cols = outlier_summary_df.loc[
        outlier_summary_df["outlier_pct"] < outlier_rate_threshold_pct, "feature"
    ].tolist()

    if not drop_candidate_cols:
        return cleaned_df

    rows_to_drop_mask = pd.Series(False, index=cleaned_df.index)
    for col in drop_candidate_cols:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows_to_drop_mask |= (cleaned_df[col] < lower) | (cleaned_df[col] > upper)

    return cleaned_df.loc[~rows_to_drop_mask].copy()
