# Queue Analytics YOLOv8

Queue Analytics is an end-to-end computer vision and data engineering pipeline designed to monitor queues and predict wait times. It utilizes YOLOv8 to track individuals in real-time, extracts meaningful data (like queue count and individual wait times), and uses XGBoost to model and predict future queue lengths based on historical data.

## Features

- **Live Stream Processing:** Connects directly to YouTube live streams (via `yt-dlp`) or uses local CCTV footage (`.mp4`) to monitor queues in real-time.
- **YOLOv8 Object Tracking:** Persistently tracks individuals across frames to accurately measure how long they stand within defined Regions of Interest (ROIs).
- **Headless Dataset Generation:** Provides a background script to continuously generate tabular dataset logs (`queue_data.csv`) from live video over long periods without rendering UI elements.
- **Wait-Time Prediction (XGBoost):** Includes a machine learning pipeline that trains on the generated CSV dataset to predict and classify wait times based on historical queue loads and time of day.
- **Gradio Interface:** A modern, interactive web UI to monitor the live feed, queue metrics, and current wait time calculations in a single dashboard.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install yt-dlp
   ```

2. **Configure ROIs:**
   Use the `boundary.py` script to define the custom polygons for your specific camera angle. This will output a `rois.json` file which the processing scripts use.
   ```bash
   python boundary.py
   ```

## Usage

**Interactive UI:**
Run the interactive dashboard to visualize the object tracking and queue metrics.
```bash
python app.py
```

**Headless Data Collection:**
Run the background dataset generator to silently watch a livestream and build your dataset.
```bash
python generate_dataset.py
```

**Train Machine Learning Model:**
Once you have collected a sizable dataset, train the XGBoost predictor.
```bash
python train_xgboost.py
```
