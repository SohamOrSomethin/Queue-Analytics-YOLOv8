import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
import os

df = pd.read_csv("queue_data.csv")

X = df.drop("actual_wait", axis=1)
y = df["actual_wait"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- REGRESSION METRICS ---")
print(f"XGBoost MAE: {mae:.2f} minutes")
print(f"XGBoost R^2: {r2:.2f}\n")

# --- ADDING CLASSIFICATION METRICS ---
# To get Accuracy and a Confusion Matrix from a Regression model, 
# we need to convert the continuous times (e.g. 12.5 mins) into categories.
def categorize_wait_time(time_in_mins):
    if time_in_mins < 5:
        return "Short Wait (<5m)"
    elif time_in_mins < 15:
        return "Medium Wait (5-15m)"
    else:
        return "Long Wait (>15m)"

# Convert actuals and predictions to categories
y_test_classes = [categorize_wait_time(t) for t in y_test]
pred_classes = [categorize_wait_time(p) for p in predictions]

accuracy = accuracy_score(y_test_classes, pred_classes)
cm = confusion_matrix(y_test_classes, pred_classes, labels=["Short Wait (<5m)", "Medium Wait (5-15m)", "Long Wait (>15m)"])

print("--- CLASSIFICATION METRICS (Categorized) ---")
print(f"Wait-Time Category Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print("Rows: Actual | Columns: Predicted")
print(cm)
print("\nDetailed Report:")
print(classification_report(y_test_classes, pred_classes, zero_division=0))

joblib.dump(model, 'xgboost_model.pkl')
print("\nModel saved to xgboost_model.pkl")