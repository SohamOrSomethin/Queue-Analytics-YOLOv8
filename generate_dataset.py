import cv2
import sys
import subprocess
import numpy as np
from datetime import datetime, timedelta
import csv
import os
from ultralytics import YOLO
import yt_dlp
from process_video import load_rois, resolve_youtube_stream, open_ffmpeg_pipe

# Philippines Standard Time = IST + 2h30m
PHT_OFFSET = timedelta(hours=2, minutes=30)




def generate_dataset_headless(yt_url, output_csv="queue_data.csv"):
    print(f"Resolving stream URL for: {yt_url}")
    stream_url, fps, width, height = resolve_youtube_stream(yt_url)

    if not stream_url:
        print("ERROR: Failed to get stream URL. Is the stream live?")
        return

    print(f"Stream URL resolved. FPS: {fps}, Resolution: {width}x{height}")

    model = YOLO("yolov8n.pt")
    queue_rois, cashier_rois = load_rois("rois.json")

    # Schema: hour, queue_size, recent_avg_wait_time, actual_wait
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","hour", "queue_size", "recent_avg_wait_time", "actual_wait"])

    print("Opening stream via ffmpeg pipe (handles YouTube URL rotation)...")
    pipe = open_ffmpeg_pipe(stream_url, width, height)
    frame_size = width * height * 3  # bytes per frame

    tracked = {}
    recent_waits = []
    frame_index = 0
    total_logged = 0

    print(f"Stream opened. FPS: {fps:.1f}. Recording to {output_csv}.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            raw = pipe.stdout.read(frame_size)

            if len(raw) < frame_size:
                print("Stream ended or frame read failed.")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame = frame.copy()   # make writable for OpenCV drawing ops

            frame_index += 1
            # Video-time timestamp — not wall-clock time.
            # Frame skip / inference lag do not affect this.
            current_time = frame_index / fps

            results = model.track(frame, persist=True, verbose=False)

            # --- Periodic status every 100 frames ---
            if frame_index % 100 == 0:
                n_detected = len(results[0].boxes) if results[0].boxes is not None else 0
                n_in_queue = sum(
                    1 for info in tracked.values() if not info["served"]
                )
                n_counted = sum(
                    1 for info in tracked.values() if info["counted"] and not info["served"]
                )
                print(
                    f"[Frame {frame_index:5d} | t={current_time:6.1f}s] "
                    f"Detected: {n_detected}  In-queue tracked: {n_in_queue}  "
                    f"Counted(≥3s): {n_counted}  Logged rows: {total_logged}"
                )
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls != 0 or box.id is None:
                        continue

                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    inside_cashier = any(
                        cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0
                        for roi in cashier_rois
                    )
                    inside_queue = any(
                        cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0
                        for roi in queue_rois
                    )

                    # ── Queue ROI path: person entered via the queue zone ──────────────
                    if inside_queue:
                        if track_id not in tracked:
                            print(f"  -> ID {track_id} entered QUEUE zone at t={current_time:.1f}s  center=({center_x},{center_y})")
                            tracked[track_id] = {
                                "enter_time": current_time,
                                "last_seen": current_time,
                                "counted": False,
                                "served": False,
                                "inside_cashier": False,
                                "via_cashier": False
                            }
                        tracked[track_id]["last_seen"] = current_time
                        tracked[track_id]["inside_cashier"] = inside_cashier

                        if current_time - tracked[track_id]["enter_time"] >= 3:
                            if not tracked[track_id]["counted"]:
                                print(f"  -> ID {track_id} now COUNTED (≥3s in queue)")
                            tracked[track_id]["counted"] = True

                        # If they've now crossed into cashier → log their queue wait
                        if (
                            inside_cashier
                            and tracked[track_id]["counted"]
                            and not tracked[track_id]["served"]
                        ):
                            wait_time = current_time - tracked[track_id]["enter_time"]
                            tracked[track_id]["served"] = True
                            total_logged += 1
                            now = datetime.now() + PHT_OFFSET
                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            hour = now.hour

                            queue_size = sum(
                                1 for info in tracked.values()
                                if info["counted"] and not info["served"]
                                and not info.get("inside_cashier", False)
                            )
                            recent_waits.append(wait_time)
                            if len(recent_waits) > 20:
                                recent_waits.pop(0)
                            recent_avg_wait = sum(recent_waits) / len(recent_waits)

                            with open(output_csv, "a", newline="") as f:
                                csv.writer(f).writerow([
                                    timestamp, hour, queue_size,
                                    round(recent_avg_wait, 2), round(wait_time, 2)
                                ])
                            print(
                                f"Logged (queue→cashier) -> Hour: {hour}  |  Queue: {queue_size}  |  "
                                f"Avg Wait: {recent_avg_wait:.1f}s  |  This Wait: {wait_time:.1f}s"
                            )

                    # ── Cashier-direct path: appeared in cashier with no prior queue entry ──
                    elif inside_cashier:
                        if track_id not in tracked:
                            print(f"  -> ID {track_id} appeared at CASHIER directly at t={current_time:.1f}s  center=({center_x},{center_y})")
                            tracked[track_id] = {
                                "enter_time": current_time,
                                "last_seen": current_time,
                                "counted": False,
                                "served": False,
                                "inside_cashier": True,
                                "via_cashier": True
                            }
                        tracked[track_id]["last_seen"] = current_time
                        tracked[track_id]["inside_cashier"] = True

                        # Count after 3s at cashier
                        if current_time - tracked[track_id]["enter_time"] >= 3:
                            if not tracked[track_id]["counted"]:
                                print(f"  -> ID {track_id} COUNTED after ≥3s at cashier")
                            tracked[track_id]["counted"] = True

            # ── Log cashier-direct people when they disappear (exit cashier) ────────
            for person_id in list(tracked.keys()):
                info = tracked[person_id]
                if current_time - info["last_seen"] > 2:
                    if info["counted"] and not info["served"] and info.get("via_cashier"):
                        wait_time = current_time - info["enter_time"]
                        info["served"] = True
                        total_logged += 1
                        now = datetime.now() + PHT_OFFSET
                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        hour = now.hour

                        queue_size = sum(
                            1 for i in tracked.values()
                            if i["counted"] and not i["served"] and not i.get("inside_cashier", False)
                        )
                        recent_waits.append(wait_time)
                        if len(recent_waits) > 20:
                            recent_waits.pop(0)
                        recent_avg_wait = sum(recent_waits) / len(recent_waits)

                        with open(output_csv, "a", newline="") as f:
                            csv.writer(f).writerow([
                                timestamp, hour, queue_size,
                                round(recent_avg_wait, 2), round(wait_time, 2)
                            ])
                        print(
                            f"Logged (cashier exit) -> Hour: {hour}  |  Queue: {queue_size}  |  "
                            f"Avg Wait: {recent_avg_wait:.1f}s  |  This Wait: {wait_time:.1f}s"
                        )
                    del tracked[person_id]
    except KeyboardInterrupt:
        print("\nData generation stopped by user.")
    finally:
        pipe.kill()


if __name__ == "__main__":
    # Usage: python generate_dataset.py
    #    or: python generate_dataset.py "https://www.youtube.com/watch?v=YOUR_ID"
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=CmtuOVxcKRo"
    generate_dataset_headless(url)
