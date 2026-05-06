"""TensorFlow neural-network training pipeline for house price prediction.

This script:
1. Loads Kaggle train/test files
2. Cleans + engineers features
3. Splits train data into train/validation
4. Grid-searches learning rate and batch size on train split
5. Compares 1/2/3-hidden-layer architectures
6. Monitors validation metrics during training
7. Saves loss curves and best submission
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import clean_test_data, clean_train_data
from src.features.features import (
    add_engineered_features,
    drop_highly_correlated_features,
    encode_categorical_features,
)

SEED = 42
TARGET_COL = "SalePrice"
EPOCHS = 80
PATIENCE = 10


@dataclass
class RunResult:
    """Training result for a single architecture + hyperparameter trial."""

    architecture_name: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    batch_size: int
    val_rmse: float
    val_mae: float
    val_r2: float
    history: dict[str, list[float]]
    model: tf.keras.Model


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def regression_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Share of predictions within 10% relative error.

    This gives an accuracy-like metric for regression monitoring.
    """
    y_true_safe = tf.maximum(tf.abs(y_true), tf.constant(1.0, dtype=tf.float32))
    rel_error = tf.abs(y_true - y_pred) / y_true_safe
    return tf.reduce_mean(tf.cast(rel_error <= 0.10, tf.float32))


def build_model(input_dim: int, hidden_layers: tuple[int, ...], lr: float) -> tf.keras.Model:
    """Build and compile a dense regression network."""
    model = tf.keras.Sequential(name=f"mlp_{len(hidden_layers)}_hidden")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[regression_accuracy, tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def prepare_features(
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data, run cleaning/feature engineering, and return split arrays."""
    train_path = base_dir / "data" / "raw" / "train.csv"
    test_path = base_dir / "data" / "raw" / "test.csv"

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    test_ids = test_raw["Id"].copy()

    train_clean = clean_train_data(train_raw)
    test_clean = clean_test_data(test_raw, train_raw)

    y = train_clean[TARGET_COL].to_numpy(dtype=np.float32)
    X_train_df = train_clean.drop(columns=[TARGET_COL])
    X_test_df = test_clean.copy()

    X_train_df = add_engineered_features(X_train_df)
    X_test_df = add_engineered_features(X_test_df)

    X_train_df = drop_highly_correlated_features(X_train_df)
    X_test_df = drop_highly_correlated_features(X_test_df)

    combined = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    combined = encode_categorical_features(combined)
    combined = combined.fillna(0)

    X_train_encoded = combined.iloc[: len(X_train_df), :].copy()
    X_test_encoded = combined.iloc[len(X_train_df) :, :].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_encoded.to_numpy(dtype=np.float32),
        y,
        test_size=0.2,
        random_state=SEED,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_encoded.to_numpy(dtype=np.float32)).astype(np.float32)

    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, test_ids.to_numpy()


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: tuple[int, ...],
    architecture_name: str,
    lr: float,
    batch_size: int,
) -> RunResult:
    """Train one model configuration and return validation metrics."""
    model = build_model(input_dim=X_train.shape[1], hidden_layers=hidden_layers, lr=lr)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping],
    )

    val_pred = model.predict(X_val, verbose=0).reshape(-1)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))
    val_r2 = float(r2_score(y_val, val_pred))

    return RunResult(
        architecture_name=architecture_name,
        hidden_layers=hidden_layers,
        learning_rate=lr,
        batch_size=batch_size,
        val_rmse=val_rmse,
        val_mae=val_mae,
        val_r2=val_r2,
        history={k: [float(vv) for vv in vals] for k, vals in history.history.items()},
        model=model,
    )


def save_loss_curve(history: dict[str, list[float]], output_path: Path) -> None:
    """Plot train/validation loss curves and save them."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Train Loss (MSE)")
    plt.plot(history["val_loss"], label="Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TensorFlow Neural Net Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run TensorFlow neural-network grid search and export best outputs."""
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parents[2]
    submissions_dir = base_dir / "reports" / "submissions"
    figures_dir = base_dir / "reports" / "figures" / "neural_network"
    metrics_dir = base_dir / "reports" / "metrics"

    submissions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val, X_test, test_ids = prepare_features(base_dir)

    architectures: dict[str, tuple[int, ...]] = {
        "1_hidden": (64,),
        "2_hidden": (128, 64),
        "3_hidden": (256, 128, 64),
    }
    learning_rates = [1e-3, 5e-4]
    batch_sizes = [32, 64]

    all_results: list[RunResult] = []
    best_result: RunResult | None = None

    for arch_name, hidden_layers in architectures.items():
        for lr in learning_rates:
            for batch_size in batch_sizes:
                result = train_and_evaluate(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    hidden_layers=hidden_layers,
                    architecture_name=arch_name,
                    lr=lr,
                    batch_size=batch_size,
                )
                all_results.append(result)
                if best_result is None or result.val_rmse < best_result.val_rmse:
                    best_result = result

    if best_result is None:
        raise RuntimeError("No training results were produced.")

    results_df = pd.DataFrame(
        [
            {
                "architecture_name": r.architecture_name,
                "hidden_layers": str(r.hidden_layers),
                "learning_rate": r.learning_rate,
                "batch_size": r.batch_size,
                "val_rmse": r.val_rmse,
                "val_mae": r.val_mae,
                "val_r2": r.val_r2,
            }
            for r in all_results
        ]
    ).sort_values("val_rmse", ascending=True)

    results_path = metrics_dir / "tensorflow_nn_gridsearch_results.csv"
    results_df.to_csv(results_path, index=False)

    best_metrics_path = metrics_dir / "tensorflow_nn_best_metrics.json"
    best_metrics_path.write_text(
        json.dumps(
            {
                "architecture_name": best_result.architecture_name,
                "hidden_layers": list(best_result.hidden_layers),
                "learning_rate": best_result.learning_rate,
                "batch_size": best_result.batch_size,
                "val_rmse": best_result.val_rmse,
                "val_mae": best_result.val_mae,
                "val_r2": best_result.val_r2,
            },
            indent=2,
        )
    )

    save_loss_curve(best_result.history, figures_dir / "tensorflow_best_loss_curve.png")

    test_preds = best_result.model.predict(X_test, verbose=0).reshape(-1)
    test_preds = np.maximum(test_preds, 0)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
    submission_path = submissions_dir / "neural_net_tensorflow_best.csv"
    submission.to_csv(submission_path, index=False)

    print("Best TensorFlow config:")
    print(f"  Architecture: {best_result.architecture_name} {best_result.hidden_layers}")
    print(f"  Learning rate: {best_result.learning_rate}")
    print(f"  Batch size: {best_result.batch_size}")
    print(f"  Validation RMSE: {best_result.val_rmse:.4f}")
    print(f"Saved submission: {submission_path}")


if __name__ == "__main__":
    main()
