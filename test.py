import warnings
import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

CSV_FILE = "queue_data_cleaned.csv"
FINAL_MODEL_FILE = "best_wait_time_model.pkl"

# Load data
df = pd.read_csv(CSV_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

if "day_of_week" not in df.columns:
    df["day_of_week"] = df["timestamp"].dt.dayofweek

# Safety: retain only your cleaned valid wait-time range
df = df[df["actual_wait"].between(3, 100)].copy()

# Sort records chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

features = ["queue_size", "hour", "day_of_week"]
target = "actual_wait"

# All unique recording dates, oldest to newest
dates = sorted(df["timestamp"].dt.date.unique())

# Need enough early dates for an initial training set
MIN_TRAINING_DATES = 4


def make_models():
    return {
        "Dummy Median": DummyRegressor(strategy="median"),

        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),

        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.3, max_iter=10000))
        ]),

        "Elastic Net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.3,
                l1_ratio=0.5,
                max_iter=10000
            ))
        ]),

        "Huber Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(
                epsilon=1.35,
                max_iter=1000
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=25,
                weights="distance"
            ))
        ]),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.2,
                gamma="scale"
            ))
        ]),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=12,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=12,
            loss="huber",
            random_state=42
        ),

        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=10,
            min_samples_leaf=15,
            l2_regularization=10.0,
            random_state=42
        ),

        "AdaBoost": AdaBoostRegressor(
            n_estimators=150,
            learning_rate=0.03,
            loss="linear",
            random_state=42
        )
    }


all_fold_results = []

# Train on older dates; test only the next date
for test_date_index in range(MIN_TRAINING_DATES, len(dates)):

    training_dates = dates[:test_date_index]
    test_date = dates[test_date_index]

    train_df = df[df["timestamp"].dt.date.isin(training_dates)]
    test_df = df[df["timestamp"].dt.date == test_date]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print("\n" + "=" * 70)
    print(f"Training through: {training_dates[-1]}")
    print(f"Testing on:       {test_date}")
    print(f"Train rows:       {len(train_df)}")
    print(f"Test rows:        {len(test_df)}")
    print("=" * 70)

    models = make_models()

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        all_fold_results.append({
            "model": model_name,
            "test_date": str(test_date),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "MAE_seconds": mean_absolute_error(y_test, prediction),
            "RMSE_seconds": mean_squared_error(
                y_test,
                prediction
            ) ** 0.5,
            "R2": r2_score(y_test, prediction)
        })

# Save results for every test day
fold_results = pd.DataFrame(all_fold_results)
fold_results.to_csv("walk_forward_each_day_results.csv", index=False)

# Calculate average metric for every model across every future day
summary = (
    fold_results
    .groupby("model", as_index=False)
    .agg(
        folds=("test_date", "count"),
        mean_MAE_seconds=("MAE_seconds", "mean"),
        mean_RMSE_seconds=("RMSE_seconds", "mean"),
        mean_R2=("R2", "mean")
    )
    .sort_values(
        by=["mean_MAE_seconds", "mean_RMSE_seconds"],
        ascending=[True, True]
    )
    .reset_index(drop=True)
)

summary.to_csv("walk_forward_model_summary.csv", index=False)

print("\n" + "=" * 70)
print("WALK-FORWARD RESULTS: RANKED BY LOWEST AVERAGE MAE")
print("=" * 70)
print(summary.to_string(index=False))

# Train the best model on all available cleaned data
best_name = summary.iloc[0]["model"]
best_model = make_models()[best_name]

best_model.fit(df[features], df[target])

joblib.dump(
    {
        "model": best_model,
        "features": features,
        "model_name": best_name,
        "validation_mae_seconds": float(
            summary.iloc[0]["mean_MAE_seconds"]
        ),
        "validation_r2": float(summary.iloc[0]["mean_R2"]),
        "max_wait_seconds": 100
    },
    FINAL_MODEL_FILE
)

print("\nBest model:", best_name)
print("Saved model:", FINAL_MODEL_FILE)
print("Per-day details: walk_forward_each_day_results.csv")
print("Model summary: walk_forward_model_summary.csv")