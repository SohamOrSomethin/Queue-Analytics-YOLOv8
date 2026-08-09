import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("queue_data.csv")

# Features are only what the vision pipeline can actually observe:
#   hour              → time of day signal
#   queue_size        → number of people currently counted in queue
#   recent_avg_wait_time → rolling average of recent actual wait times (context signal)
FEATURES = ["hour", "queue_size", "recent_avg_wait_time"]
TARGET = "actual_wait"

# Validate that the CSV has the expected columns
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns in queue_data.csv: {missing}\n"
        f"Expected: {FEATURES + [TARGET]}\n"
        f"Got: {df.columns.tolist()}\n"
        "Re-run generate_dataset.py to collect data with the updated schema."
    )

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- REGRESSION METRICS ---")
print(f"XGBoost MAE: {mae:.2f} seconds")
print(f"XGBoost R^2: {r2:.2f}")

joblib.dump(model, 'xgboost_model.pkl')
print("\nModel saved to xgboost_model.pkl")