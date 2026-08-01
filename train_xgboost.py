import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("queue_data.csv")

# FIX 4: Removed 'party_size' — we cannot detect parties, only individuals.
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
print(f"XGBoost R^2: {r2:.2f}\n")

# --- CLASSIFICATION METRICS ---
# Bin continuous wait times into categories for interpretability
def categorize_wait_time(time_in_secs):
    if time_in_secs < 60:
        return "Short (<1m)"
    elif time_in_secs < 300:
        return "Medium (1-5m)"
    else:
        return "Long (>5m)"

y_test_classes = [categorize_wait_time(t) for t in y_test]
pred_classes = [categorize_wait_time(p) for p in predictions]

accuracy = accuracy_score(y_test_classes, pred_classes)
labels = ["Short (<1m)", "Medium (1-5m)", "Long (>5m)"]
cm = confusion_matrix(y_test_classes, pred_classes, labels=labels)

print("--- CLASSIFICATION METRICS (Categorized) ---")
print(f"Wait-Time Category Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print("Rows: Actual | Columns: Predicted")
print(cm)
print("\nDetailed Report:")
print(classification_report(y_test_classes, pred_classes, zero_division=0))

joblib.dump(model, 'xgboost_model.pkl')
print("\nModel saved to xgboost_model.pkl")