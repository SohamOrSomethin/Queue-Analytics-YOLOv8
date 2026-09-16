import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DATA_FILE = "queue_data.csv"
MODEL_FILE = "xgboost_model.pkl"
METRICS_FILE = "model_metrics.json"

FEATURE_COLUMNS = [
    "hour",
    "queue_size",
    "recent_avg_wait_time",
]

TARGET_COLUMN = "actual_wait"

MIN_VALID_WAIT_SECONDS = 1
MAX_VALID_WAIT_SECONDS = 200


def load_and_clean_data(csv_path):
    """
    Loads queue_data.csv and applies basic data cleaning.

    Expected columns:
    timestamp,hour,queue_size,recent_avg_wait_time,actual_wait

    Wait-time unit: seconds.
    Rows with actual_wait > 200 seconds are considered invalid.
    """

    df = pd.read_csv(csv_path)

    required_columns = [
        "timestamp",
        "hour",
        "queue_size",
        "recent_avg_wait_time",
        "actual_wait",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_path}: {missing_columns}"
        )

    print(f"\n[INFO] Raw dataset rows: {len(df)}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    numeric_columns = [
        "hour",
        "queue_size",
        "recent_avg_wait_time",
        "actual_wait",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    before_missing_cleanup = len(df)

    df = df.dropna(
        subset=[
            "timestamp",
            "hour",
            "queue_size",
            "recent_avg_wait_time",
            "actual_wait",
        ]
    )

    print(
        "[INFO] Removed rows with missing/invalid values: "
        f"{before_missing_cleanup - len(df)}"
    )

    before_hour_cleanup = len(df)

    df = df[
        (df["hour"] >= 0) &
        (df["hour"] <= 23)
    ]

    print(
        "[INFO] Removed rows with invalid hour values: "
        f"{before_hour_cleanup - len(df)}"
    )

    before_queue_cleanup = len(df)

    df = df[
        (df["queue_size"] >= 0) &
        (df["recent_avg_wait_time"] >= 0)
    ]

    print(
        "[INFO] Removed rows with invalid queue/wait features: "
        f"{before_queue_cleanup - len(df)}"
    )

    before_wait_cleanup = len(df)

    df = df[
        (df["actual_wait"] >= MIN_VALID_WAIT_SECONDS) &
        (df["actual_wait"] <= MAX_VALID_WAIT_SECONDS)
    ]

    print(
        "[INFO] Removed invalid actual_wait rows "
        f"(< {MIN_VALID_WAIT_SECONDS}s or > {MAX_VALID_WAIT_SECONDS}s): "
        f"{before_wait_cleanup - len(df)}"
    )

    df = df.drop_duplicates(
        subset=[
            "timestamp",
            "hour",
            "queue_size",
            "recent_avg_wait_time",
            "actual_wait",
        ]
    )

    print(f"[INFO] Clean dataset rows: {len(df)}")

    if len(df) < 30:
        raise ValueError(
            "Not enough valid rows after cleaning. "
            "Collect more queue wait-time records."
        )

    return df


def train_xgboost_model(X_train, y_train):
    """
    Creates and trains the XGBoost regression model.
    """

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def calculate_metrics(y_true, y_pred):
    """
    Returns common regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae_seconds": round(float(mae), 2),
        "rmse_seconds": round(float(rmse), 2),
        "r2_score": round(float(r2), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for queue waiting-time prediction."
    )

    parser.add_argument(
        "--data",
        default=DATA_FILE,
        help="Path to queue CSV dataset."
    )

    parser.add_argument(
        "--model-out",
        default=MODEL_FILE,
        help="Path for storing the trained model."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing."
    )

    args = parser.parse_args()

    csv_path = Path(args.data)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = load_and_clean_data(csv_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    print(f"\n[INFO] Training rows: {len(X_train)}")
    print(f"[INFO] Testing rows: {len(X_test)}")

    model = train_xgboost_model(X_train, y_train)

    xgb_predictions = model.predict(X_test)

    baseline_prediction = np.full(
        shape=len(y_test),
        fill_value=y_train.mean(),
        dtype=float,
    )

    xgb_metrics = calculate_metrics(
        y_test,
        xgb_predictions,
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_prediction,
    )

    improvement = (
        (baseline_metrics["mae_seconds"] - xgb_metrics["mae_seconds"])
        / baseline_metrics["mae_seconds"]
    ) * 100

    print("\n" + "=" * 60)
    print("WAIT-TIME PREDICTION MODEL RESULTS")
    print("=" * 60)

    print("\nXGBoost Model")
    print(
        f"MAE: {xgb_metrics['mae_seconds']} seconds"
    )
    print(
        f"RMSE: {xgb_metrics['rmse_seconds']} seconds"
    )
    print(
        f"R² Score: {xgb_metrics['r2_score']}"
    )

    print("\nMean Baseline")
    print(
        f"MAE: {baseline_metrics['mae_seconds']} seconds"
    )
    print(
        f"RMSE: {baseline_metrics['rmse_seconds']} seconds"
    )

    print(
        f"\nXGBoost MAE improvement over baseline: "
        f"{improvement:.2f}%"
    )

    print("\nFeature Importance")

    importance_pairs = sorted(
        zip(
            FEATURE_COLUMNS,
            model.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature_name, importance in importance_pairs:
        print(
            f"{feature_name}: {importance:.4f}"
        )

    model_payload = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "wait_time_unit": "seconds",
        "min_valid_wait_seconds": MIN_VALID_WAIT_SECONDS,
        "max_valid_wait_seconds": MAX_VALID_WAIT_SECONDS,
    }

    with open(args.model_out, "wb") as model_file:
        pickle.dump(
            model_payload,
            model_file,
        )

    metrics_payload = {
        "dataset_file": str(csv_path),
        "raw_rows": int(len(pd.read_csv(csv_path))),
        "clean_rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "wait_time_unit": "seconds",
        "invalid_wait_rule": (
            f"actual_wait < {MIN_VALID_WAIT_SECONDS} "
            f"or actual_wait > {MAX_VALID_WAIT_SECONDS}"
        ),
        "xgboost_metrics": xgb_metrics,
        "baseline_metrics": baseline_metrics,
        "mae_improvement_percent": round(
            float(improvement),
            2,
        ),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as metrics_file:
        json.dump(
            metrics_payload,
            metrics_file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print(
        f"[SUCCESS] Model saved to: {args.model_out}"
    )
    print(
        f"[SUCCESS] Metrics saved to: {METRICS_FILE}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()