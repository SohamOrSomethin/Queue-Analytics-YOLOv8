import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset: records above 100 seconds are already removed
df = pd.read_csv("queue_data_cleaned.csv")

# Convert timestamp and derive weekday
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["day_of_week"] = df["timestamp"].dt.dayofweek

features = ["queue_size", "hour", "day_of_week"]
target = "actual_wait"

# Chronological evaluation: newest three dates form the test set
dates = sorted(df["timestamp"].dt.date.unique())
test_dates = dates[-3:]

train = df[~df["timestamp"].dt.date.isin(test_dates)].copy()
test = df[df["timestamp"].dt.date.isin(test_dates)].copy()

model = RandomForestRegressor(
    n_estimators=600,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=-1
)

model.fit(train[features], train[target])

predictions = model.predict(test[features])

mae = mean_absolute_error(test[target], predictions)
rmse = mean_squared_error(test[target], predictions) ** 0.5
r2 = r2_score(test[target], predictions)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.3f}")

# Refit using all cleaned data before deployment
final_model = RandomForestRegressor(
    n_estimators=600,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(df[features], df[target])

joblib.dump(final_model, "final_wait_time_model.pkl")
print("Saved: final_wait_time_model.pkl")