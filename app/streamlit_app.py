"""Simple Streamlit chatbot app for Bayesian house-price prediction."""

# from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sklearn.base import BaseEstimator, TransformerMixin

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.preprocess import clean_test_data
from src.features.features import add_engineered_features, drop_highly_correlated_features


class SelectiveLogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transform to selected numeric columns."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SelectiveLogTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                min_val = X[col].min()
                if pd.notna(min_val) and min_val <= -1:
                    X[col] = X[col] + abs(min_val) + 1.0
                X[col] = np.log1p(X[col])
        return X


CACHE_PATH = BASE_DIR / "app" / "cache" / "bayesian_defaults.json"
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_CANDIDATES = [
    BASE_DIR / "bestmodels" / "bayesian_ridge.joblib",
    BASE_DIR / "reports" / "bestmodels" / "bayesian_ridge.joblib",
    BASE_DIR / "reports" / "bestmodels" / "bayesian_ridge_probabilistic.joblib",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No Bayesian model found. Checked: {[str(p) for p in MODEL_CANDIDATES]}"
    )


def _extract_json_string(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return stripped[start : end + 1]


def _cast_value(value: Any, dtype_text: str) -> Any:
    if value is None:
        return None

    dtype_lower = dtype_text.lower()

    if "int" in dtype_lower:
        if isinstance(value, str):
            value = value.strip()
        return int(float(value))
    if "float" in dtype_lower:
        if isinstance(value, str):
            value = value.strip()
        return float(value)
    return str(value).strip()


def _normalize_extracted(
    extracted: dict[str, Any], raw_defaults: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw_value in extracted.items():
        if key not in raw_defaults:
            continue
        try:
            normalized[key] = _cast_value(raw_value, raw_defaults[key]["dtype"])
        except (ValueError, TypeError):
            normalized[key] = None
    return normalized


def _build_full_raw_feature_map(
    extracted: dict[str, Any], raw_defaults: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    full_map: dict[str, Any] = {}
    for key, spec in raw_defaults.items():
        value = extracted.get(key)
        if value is None or value == "":
            value = spec.get("value")
        full_map[key] = _cast_value(value, spec["dtype"]) if value is not None else None
    return full_map


def _prepare_model_input(
    full_raw_map: dict[str, Any],
    model_defaults: dict[str, dict[str, Any]],
    train_df_raw: pd.DataFrame,
) -> pd.DataFrame:
    raw_row = pd.DataFrame([full_raw_map])
    cleaned = clean_test_data(raw_row, train_df_raw)
    engineered = add_engineered_features(cleaned)
    engineered = drop_highly_correlated_features(engineered)

    for col, spec in model_defaults.items():
        if col not in engineered.columns:
            engineered[col] = spec["value"]
        if engineered[col].isna().any():
            engineered[col] = engineered[col].fillna(spec["value"])

    return engineered


def _predict_with_uncertainty(model: Any, model_input_df: pd.DataFrame) -> tuple[float, float, float]:
    prediction = model.predict(model_input_df, return_std=True)

    if isinstance(prediction, tuple) and len(prediction) == 2:
        mu_log, sigma_log = prediction
        mu_log_scalar = float(np.asarray(mu_log).ravel()[0])
        sigma_log_scalar = max(float(np.asarray(sigma_log).ravel()[0]), 1e-8)
        expected_price = max(float(np.expm1(mu_log_scalar)), 0.0)
        lower = max(float(np.expm1(mu_log_scalar - 1.96 * sigma_log_scalar)), 0.0)
        upper = max(float(np.expm1(mu_log_scalar + 1.96 * sigma_log_scalar)), 0.0)
        return expected_price, lower, upper

    pred_scalar = float(np.asarray(prediction).ravel()[0])
    expected_price = max(float(np.expm1(pred_scalar)), 0.0)
    margin = expected_price * 0.1
    return expected_price, max(expected_price - margin, 0.0), expected_price + margin


def _call_groq_feature_extractor(
    client: Groq,
    user_text: str,
    raw_defaults: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature_specs = {
        k: {"dtype": v["dtype"], "default": v["value"]} for k, v in raw_defaults.items()
    }

    system_prompt = (
        "You extract structured real-estate features from user messages.\n"
        "Extract only features from the provided feature list.\n"
        "Infer features even when implied (implicit values).\n"
        "Return only one valid JSON object and no extra text.\n"
        "Keys must be exact feature names. Do not invent keys.\n"
        "If a feature is not mentioned or cannot be inferred, omit that key."
    )
    user_prompt = (
        f"Feature specs:\n{json.dumps(feature_specs)}\n\n"
        f"User message:\n{user_text}\n\n"
        "Return JSON only."
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(_extract_json_string(content))
    if not isinstance(payload, dict):
        raise ValueError("Extractor response is not a JSON object.")
    return payload


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    st.set_page_config(page_title="RealEstateConsultant", page_icon="🏠", layout="centered")
    st.title("RealEstateConsultant")
    st.caption("Chat-based house price estimate using Groq Llama 8B + Bayesian Ridge")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe the property details...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("Missing GROQ_API_KEY in .env")

            payload = _load_json(CACHE_PATH)
            raw_defaults = payload["raw_input_defaults"]
            model_defaults = payload["model_input_defaults"]
            train_df_raw = pd.read_csv(TRAIN_PATH)
            model = joblib.load(_find_model_path())
            groq_client = Groq(api_key=groq_api_key)

            extracted_raw = _call_groq_feature_extractor(groq_client, user_input, raw_defaults)
            normalized_extracted = _normalize_extracted(extracted_raw, raw_defaults)
            full_raw_map = _build_full_raw_feature_map(normalized_extracted, raw_defaults)
            model_input_df = _prepare_model_input(full_raw_map, model_defaults, train_df_raw)
            expected_price, lower, upper = _predict_with_uncertainty(model, model_input_df)
            margin = max(upper - expected_price, expected_price - lower)

            response_text = (
                f"The expected house price is {_format_currency(expected_price - margin)} "
                f"to {_format_currency(expected_price + margin)}."
            )
            st.markdown(response_text)

            with st.expander("Extracted Features JSON"):
                st.json(normalized_extracted)
            with st.expander("Final Features Sent to Model (JSON)"):
                st.json(full_raw_map)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as exc:
            error_text = f"Could not generate prediction: {exc}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})


if __name__ == "__main__":
    main()
