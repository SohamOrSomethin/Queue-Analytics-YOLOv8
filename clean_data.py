import pandas as pd

df = pd.read_csv("queue_data.csv")

df_clean = df[
    df["actual_wait"].between(3.5, 100, inclusive="both")
].copy()

# Put the timestamp code HERE
df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

# You already have an hour column, so do NOT add this line:
# df_clean["hour"] = df_clean["timestamp"].dt.hour

df_clean["day_of_week"] = df_clean["timestamp"].dt.dayofweek

print(df_clean[
    ["timestamp", "hour", "day_of_week", "queue_size", "actual_wait"]
].head())

# Save only after adding day_of_week
df_clean.to_csv("queue_data_cleaned.csv", index=False)