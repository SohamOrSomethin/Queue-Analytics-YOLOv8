import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "hour",
    "queue_size",
    "recent_avg_wait_time",
]

TARGET_COLUMN = "actual_wait"

MIN_VALID_WAIT_SECONDS = 1
MAX_VALID_WAIT_SECONDS = 200


def load_clean_data(csv_path):
    """
    Loads and cleans the queue dataset.

    Invalid rows include:
    - Missing values
    - Negative queue features
    - Invalid hour
    - actual_wait below 1 second
    - actual_wait above 200 seconds
    """

    df = pd.read_csv(csv_path)

    required_columns = [
        "timestamp",
        "hour",
        "queue_size",
        "recent_avg_wait_time",
        "actual_wait",
    ]

    df = df[required_columns].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    df = df[
        (df["hour"] >= 0) &
        (df["hour"] <= 23) &
        (df["queue_size"] >= 0) &
        (df["recent_avg_wait_time"] >= 0) &
        (df["actual_wait"] >= MIN_VALID_WAIT_SECONDS) &
        (df["actual_wait"] <= MAX_VALID_WAIT_SECONDS)
    ]

    return df


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "r2": r2_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline models with XGBoost."
    )

    parser.add_argument(
        "--data",
        default="queue_data.csv",
        help="Dataset CSV path."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Testing split size."
    )

    args = parser.parse_args()

    df = load_clean_data(args.data)

    print(
        f"[INFO] Valid rows after cleaning: {len(df)}"
    )

    if len(df) < 30:
        raise ValueError(
            "Not enough clean records for evaluation."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    mean_prediction = np.full(
        len(y_test),
        y_train.mean(),
    )

    linear_model = LinearRegression()
    linear_model.fit(
        X_train,
        y_train,
    )

    linear_prediction = linear_model.predict(
        X_test
    )

    xgb_model = XGBRegressor(
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

    xgb_model.fit(
        X_train,
        y_train,
    )

    xgb_prediction = xgb_model.predict(
        X_test
    )

    results = {
        "Mean Baseline": metrics(
            y_test,
            mean_prediction,
        ),
        "Linear Regression": metrics(
            y_test,
            linear_prediction,
        ),
        "XGBoost": metrics(
            y_test,
            xgb_prediction,
        ),
    }

    print("\n" + "=" * 63)
    print(
        f"{'MODEL':<22}"
        f"{'MAE (sec)':>13}"
        f"{'RMSE (sec)':>14}"
        f"{'R²':>10}"
    )
    print("=" * 63)

    for model_name, result in results.items():
        print(
            f"{model_name:<22}"
            f"{result['mae']:>13.2f}"
            f"{result['rmse']:>14.2f}"
            f"{result['r2']:>10.4f}"
        )

    print("=" * 63)

    baseline_mae = results["Mean Baseline"]["mae"]
    xgb_mae = results["XGBoost"]["mae"]

    improvement = (
        (baseline_mae - xgb_mae)
        / baseline_mae
    ) * 100

    print(
        f"\nXGBoost MAE improvement over "
        f"Mean Baseline: {improvement:.2f}%"
    )

    print(
        "\nNote: All records with actual_wait "
        "> 200 seconds were removed as invalid."
    )


if __name__ == "__main__":
    main()