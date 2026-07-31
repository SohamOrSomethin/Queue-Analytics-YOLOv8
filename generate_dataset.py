import cv2
import time
from datetime import datetime
import csv
import os
from ultralytics import YOLO
import yt_dlp
from process_video import load_rois

def resolve_youtube_stream(yt_url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(yt_url, download=False)
        return info_dict.get('url', None)

def generate_dataset_headless(yt_url, output_csv="queue_data.csv"):
    stream_url = resolve_youtube_stream(yt_url)
    if not stream_url:
        print("Failed to get stream URL.")
        return

    model = YOLO("yolov8n.pt")
    queue_roi, cashier_rois = load_rois("rois.json")

    # Ensure CSV has headers if it doesn't exist
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["queue_count", "hour", "actual_wait"])

    cap = cv2.VideoCapture(stream_url)
    tracked = {}
    print(f"Starting headless data generation to {output_csv}.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Stream ended or error reading frame.")
                break

            current_time = time.time()
            results = model.track(frame, persist=True, verbose=False)
            count = 0

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls != 0 or box.id is None:
                        continue
                    
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    inside_cashier = any(cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0 for roi in cashier_rois)
                    inside_queue = cv2.pointPolygonTest(queue_roi, (center_x, center_y), False) >= 0

                    if inside_cashier and track_id in tracked and tracked[track_id]["counted"] and not tracked[track_id]["served"]:
                        wait_time = current_time - tracked[track_id]["enter_time"]
                        tracked[track_id]["served"] = True
                        hour = datetime.now().hour
                        
                        # Write row to CSV
                        with open(output_csv, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([count, hour, round(wait_time, 2)])
                        
                        print(f"Logged Data Point -> Queue Size: {count}, Hour: {hour}, Wait Time: {wait_time:.2f}s")

                    if inside_queue:
                        if track_id not in tracked:
                            tracked[track_id] = {"enter_time": current_time, "last_seen": current_time, "counted": False, "served": False}
                        tracked[track_id]["last_seen"] = current_time
                        if current_time - tracked[track_id]["enter_time"] >= 3:
                            tracked[track_id]["counted"] = True

            for person_id in list(tracked.keys()):
                if current_time - tracked[person_id]["last_seen"] > 2:
                    del tracked[person_id]
            
            count = sum(info["counted"] for info in tracked.values())

    except KeyboardInterrupt:
        print("Data generation stopped.")
    finally:
        cap.release()

if __name__ == "__main__":
    generate_dataset_headless("https://www.youtube.com/watch?v=CmtuOVxcKRo")
