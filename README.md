# Queue Analytics YOLOv8

Queue Analytics is an end-to-end computer vision and data engineering pipeline designed to monitor queues and predict wait times. It uses YOLOv8 to track individuals in real-time, extracts meaningful data (queue count and individual wait times), and uses XGBoost to predict future wait times based on historical data.

## How It Works

1. **YOLOv8** detects and persistently tracks people across video frames
2. **ROIs (Regions of Interest)** tell the system which part of the frame is the queue area and which is the cashier — without this, every person in view would be counted
3. When a tracked person moves from the queue ROI into the cashier ROI, their wait time is recorded
4. This data is logged to `queue_data.csv` and used to train an XGBoost model that predicts wait times based on hour of day and current queue size

## Features

- **Live Stream Processing:** Connects directly to YouTube live streams (via `yt-dlp` + `imageio-ffmpeg`) or local CCTV footage — no manual ffmpeg installation required
- **YOLOv8 Object Tracking:** Persistently tracks individuals across frames for accurate wait time measurement
- **Cashier Zone Exclusion:** People at the cashier are excluded from the queue count to avoid double-counting
- **Headless Dataset Generation:** Background script continuously logs real wait times to `queue_data.csv`
- **Wait-Time Prediction (XGBoost):** Trains on the generated CSV to predict wait times from queue size and time of day
- **Gradio Interface:** Interactive web UI to draw ROIs, monitor the live feed, and view queue metrics

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything including `imageio-ffmpeg`, which bundles its own ffmpeg binary. **You do not need to install ffmpeg separately or add it to your PATH.**

### 2. Draw your ROIs

Launch the Gradio app and use the **Step 1: Setup ROIs** tab to draw your queue and cashier zones on your camera's first frame. Save them to `rois.json`.

```bash
python app.py
```

- **Queue ROI** — the area where people wait in line (draw a polygon around the waiting zone)
- **Cashier ROI** — tight around just the counter/register (must not fully contain the queue ROI)

> **Why manual ROIs?** YOLO detects people, not queue zones. The ROI is the business logic layer — it tells the system which part of the scene belongs to your specific store layout. This means the same code works for any camera angle without retraining.

## Generating a Dataset

This is the most important step. The model is only as good as the data you collect from your real stream.

### Step 1 — Draw ROIs (if not done already)

```bash
python app.py
```

Go to **Step 1: Setup ROIs** → load first frame → draw polygons → **Save All ROIs to rois.json**.

### Step 2 — Run the headless recorder during open hours

```bash
python generate_dataset.py "https://www.youtube.com/watch?v=YOUR_STREAM_ID"
```

The script will print a line every time a customer is served:
```
Logged -> Hour: 12  |  Queue: 3  |  Avg Wait: 45.2s  |  This Wait: 51.0s
```

Each logged line is one row in `queue_data.csv`. Let it run for **at least 2–3 hours during peak times**. Aim for **200+ rows** before training.

Press `Ctrl+C` to stop.

### Step 3 — Train the model

```bash
python train_xgboost.py
```

This trains an XGBoost regressor on `["hour", "queue_size", "recent_avg_wait_time"]` to predict `actual_wait`. It prints MAE, R², and a classification breakdown.

## CSV Schema

`queue_data.csv` uses the following columns:

| Column | Description |
|---|---|
| `timestamp` | Wall-clock datetime when the customer was served |
| `hour` | Hour of day (0–23) when the customer was served |
| `queue_size` | Number of people counted in queue at that moment |
| `recent_avg_wait_time` | Rolling average of the last 20 recorded wait times (seconds) |
| `actual_wait` | Measured wait time for this customer (seconds, video-time based) |

## Usage

**Interactive Monitor:**
```bash
python app.py
```

**Headless Data Collection:**
```bash
python generate_dataset.py "https://www.youtube.com/watch?v=YOUR_STREAM_ID"
```

**Train ML Model:**
```bash
python train_xgboost.py
```
