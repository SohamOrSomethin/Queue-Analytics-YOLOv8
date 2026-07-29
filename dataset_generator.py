import pandas as pd
import random
from datetime import datetime, timedelta


def generate_mock_queue_data(start_time, end_time):
    current_time = start_time
    data = []
    
    # Keep track of recent actual wait times to simulate a rolling average
    recent_waits = [2.0, 2.5, 3.0]

    while current_time < end_time:
        party_size = random.randint(1, 6)
        queue_size = random.randint(0, 10)
        
        # Calculate recent_avg_wait_time
        recent_avg_wait_time = round(sum(recent_waits) / len(recent_waits), 2)
        
        # Simulate actual wait time in minutes (affected by queue_size)
        base_wait = random.uniform(1.0, 3.0)
        actual_wait = round(base_wait + (queue_size * 0.5), 2)
        
        # Update our running list of recent waits (keep last 3)
        recent_waits.append(actual_wait)
        recent_waits.pop(0)

        data.append({
            "hour": current_time.hour,
            "party_size": party_size,
            "queue_size": queue_size,
            "recent_avg_wait_time": recent_avg_wait_time,
            "actual_wait": actual_wait
        })

        # Increment time by a random interval
        current_time += timedelta(seconds=random.randint(5, 60))

    return pd.DataFrame(data)

start = datetime(2024, 1, 1, 9, 0, 0)
end = datetime(2024, 1, 1, 17, 0, 0)

df = generate_mock_queue_data(start, end)

df.to_csv("queue_data.csv", index=False)
print(df.head())