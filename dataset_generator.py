import pandas as pd
import random
from datetime import datetime, timedelta

def get_time_multiplier(hour):
    """Returns a multiplier based on the time of day (assuming a typical restaurant)."""
    if 9 <= hour < 11:
        return 0.2  # Morning: very low wait
    elif 11 <= hour < 14:
        return 1.5  # Lunch rush: high wait
    elif 14 <= hour < 17:
        return 0.5  # Tea time / mid-afternoon: low wait
    elif 17 <= hour <= 22:
        return 2.0  # Dinner rush: peak wait
    else:
        return 0.1  # Late night / outside hours

def generate_mock_queue_data(start_time, end_time):
    current_time = start_time
    data = []
    
    # Keep track of recent actual wait times to simulate a rolling average
    recent_waits = [2.0, 2.5, 3.0]

    while current_time < end_time:
        hour = current_time.hour
        multiplier = get_time_multiplier(hour)
        
        party_size = random.randint(1, 6)
        
        # Queue size is naturally larger during peak hours
        queue_size = int(random.randint(0, 8) * multiplier)
        
        # Calculate recent_avg_wait_time
        recent_avg_wait_time = round(sum(recent_waits) / len(recent_waits), 2)
        
        # Base wait time is also scaled by the time multiplier (kitchen gets backed up)
        base_wait = random.uniform(1.0, 3.0) * multiplier
        
        # Larger parties add slight extra wait time
        party_penalty = (party_size - 2) * 0.5 if party_size > 2 else 0
        
        actual_wait = round(base_wait + (queue_size * 0.8) + party_penalty, 2)
        actual_wait = max(0, actual_wait) # Ensure no negative wait times
        
        # Update our running list of recent waits (keep last 3)
        recent_waits.append(actual_wait)
        recent_waits.pop(0)

        data.append({
            "hour": hour,
            "party_size": party_size,
            "queue_size": queue_size,
            "recent_avg_wait_time": recent_avg_wait_time,
            "actual_wait": actual_wait
        })

        # Increment time by a random interval
        current_time += timedelta(seconds=random.randint(10, 60))

    return pd.DataFrame(data)

# Run a full day from 9 AM to 10 PM
start = datetime(2024, 1, 1, 9, 0, 0)
end = datetime(2024, 1, 1, 22, 0, 0)

df = generate_mock_queue_data(start, end)

df.to_csv("queue_data.csv", index=False)
print("Generated new queue_data.csv with time-of-day variations!")
print(df.head())