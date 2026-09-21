import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Change this filename if your clean_data.py saved a different name
df = pd.read_csv("queue_data_cleaned.csv")

target = "actual_wait"

feature_sets = {
    "without_recent_avg": [
        "queue_size",
        "hour",
        "day_of_week"
    ],
    "with_recent_avg": [
        "queue_size",
        "hour",
        "day_of_week",
        "recent_avg_wait_time"
    ]
}

# One fixed split, shared by both models, so the comparison is fair
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

results = []

for name, features in feature_sets.items():
    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "model": name,
        "features": ", ".join(features),
        "MAE_seconds": round(mae, 2),
        "RMSE_seconds": round(rmse, 2),
        "R2": round(r2, 3)
    })

    joblib.dump(model, f"{name}_model.pkl")

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved:")
print("- without_recent_avg_model.pkl")
print("- with_recent_avg_model.pkl")
print("- model_comparison.csv")