"""Preprocessing helpers for train/test cleaning."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def is_categorical_series(series: pd.Series) -> bool:
    """Return True when a series should be treated as categorical."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    )


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Get column names considered categorical for a dataframe."""
    return [col for col in df.columns if is_categorical_series(df[col])]


def _fill_base_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared missing-value rules used for both train and test."""
    df = df.copy()

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    na_means_none = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]
    for col in na_means_none:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    if "GarageType" in df.columns:
        df["GarageType"] = df["GarageType"].fillna("None")

    if "MasVnrArea" in df.columns and "MasVnrType" in df.columns:
        mas_none = df["MasVnrType"].isna() | (df["MasVnrType"] == "None")
        mas_mean = df["MasVnrArea"].mean()
        df.loc[mas_none, "MasVnrArea"] = df.loc[mas_none, "MasVnrArea"].fillna(0)
        df.loc[~mas_none, "MasVnrArea"] = df.loc[~mas_none, "MasVnrArea"].fillna(
            mas_mean
        )

    garage_numeric_cols = ["GarageYrBlt", "GarageArea", "GarageCars"]
    if "GarageType" in df.columns:
        no_garage = df["GarageType"].isna() | (df["GarageType"] == "None")
        for col in garage_numeric_cols:
            if col in df.columns:
                median_value = df[col].median()
                df.loc[no_garage, col] = df.loc[no_garage, col].fillna(0)
                df.loc[~no_garage, col] = df.loc[~no_garage, col].fillna(
                    median_value
                )

    garage_categorical_cols = ["GarageFinish", "GarageQual", "GarageCond"]
    if "GarageType" in df.columns:
        no_garage = df["GarageType"].isna() | (df["GarageType"] == "None")
        for col in garage_categorical_cols:
            if col in df.columns:
                df.loc[no_garage, col] = df.loc[no_garage, col].fillna("None")

    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    return df


def clean_train_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Clean training data using the shared missing-value rules. eda_cleaning.ipynb"""
    return _fill_base_missing_values(train_df)


def _fill_test_numeric_from_train(
    test_df: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    """Fill numeric missing values in test using train medians."""
    df = test_df.copy()
    numeric_missing_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any()
    ]
    for col in numeric_missing_cols:
        if col in train_df.columns:
            df[col] = df[col].fillna(train_df[col].median())
    return df


def _fill_test_categorical_from_train(
    test_df: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    """Fill categorical missing values in test using train modes and exceptions."""
    df = test_df.copy()
    categorical_missing_cols = [
        col
        for col in df.columns
        if is_categorical_series(df[col]) and df[col].isna().any()
    ]
    for col in categorical_missing_cols:
        if col in ["Exterior1st", "Exterior2nd"]:
            df[col] = df[col].fillna("Other")
        elif col == "SaleType":
            df[col] = df[col].fillna("Oth")
        elif col in train_df.columns:
            train_mode = train_df[col].mode(dropna=True)
            fill_value = train_mode.iloc[0] if not train_mode.empty else "None"
            df[col] = df[col].fillna(fill_value)
    return df


def clean_test_data(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Clean test/validation data using train-derived imputations. eda_cleaning.ipynb"""
    df = _fill_base_missing_values(test_df)
    df = _fill_test_numeric_from_train(df, train_df)
    df = _fill_test_categorical_from_train(df, train_df)
    return df
