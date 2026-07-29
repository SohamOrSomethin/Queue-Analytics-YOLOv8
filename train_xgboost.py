import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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

print(f"XGBoost MAE: {mae:.2f} minutes")
print(f"XGBoost R^2: {r2:.2f}")

joblib.dump(model, 'xgboost_model.pkl')
print("Model saved to xgboost_model.pkl")