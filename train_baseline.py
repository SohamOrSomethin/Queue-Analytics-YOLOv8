import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv('queue_data.csv')

X = df.drop(columns=['actual_wait'])
y = df['actual_wait']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Baseline Random Forest MAE: {mae:.2f} minutes")
print(f"Baseline Random Forest R^2: {r2:.2f}")

joblib.dump(model, 'baseline_rf_model.pkl')
print("Model saved to baseline_rf_model.pkl")