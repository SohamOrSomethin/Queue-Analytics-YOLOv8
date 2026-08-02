import cv2
import sys
import subprocess
import numpy as np
from datetime import datetime
import csv
import os
from ultralytics import YOLO
import yt_dlp
from process_video import load_rois


def resolve_youtube_stream(yt_url):
    """
    Use yt-dlp to extract stream metadata.
    Returns (stream_url, fps, width, height).
    """
    ydl_opts = {'format': 'best[ext=mp4]/best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_url, download=False)
        stream_url = info.get('url', None)
        fps = float(info.get('fps') or 30)
        width = int(info.get('width') or 1280)
        height = int(info.get('height') or 720)
        return stream_url, fps, width, height


def open_ffmpeg_pipe(stream_url, width, height):
    """
    Pipe stream frames through a system ffmpeg subprocess.

    Why: OpenCV's internal ffmpeg cannot handle YouTube CDN URL rotation mid-stream.
    When the CDN URL expires it throws a TLS error and the capture dies.
    System ffmpeg handles m3u8 playlists natively — it fetches new segment URLs
    automatically and never drops the stream.
    """
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',       # suppress ffmpeg noise
        '-i', stream_url,
        '-f', 'rawvideo',           # output raw pixel data
        '-pix_fmt', 'bgr24',        # OpenCV expects BGR
        '-vf', f'scale={width}:{height}',
        'pipe:1'                    # write to stdout
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


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
            writer.writerow(["hour", "queue_size", "recent_avg_wait_time", "actual_wait"])

    print("Opening stream via ffmpeg pipe (handles YouTube URL rotation)...")
    pipe = open_ffmpeg_pipe(stream_url, width, height)
    frame_size = width * height * 3  # bytes per frame

    tracked = {}
    recent_waits = []
    frame_index = 0

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

                    # Person reached cashier → record their wait time
                    if (
                        inside_cashier
                        and track_id in tracked
                        and tracked[track_id]["counted"]
                        and not tracked[track_id]["served"]
                    ):
                        wait_time = current_time - tracked[track_id]["enter_time"]
                        tracked[track_id]["served"] = True
                        hour = datetime.now().hour

                        # Current queue size (excludes people at cashier)
                        queue_size = sum(
                            1 for info in tracked.values()
                            if info["counted"]
                            and not info["served"]
                            and not info.get("inside_cashier", False)
                        )

                        recent_waits.append(wait_time)
                        if len(recent_waits) > 20:
                            recent_waits.pop(0)
                        recent_avg_wait = sum(recent_waits) / len(recent_waits)

                        with open(output_csv, "a", newline="") as f:
                            csv.writer(f).writerow([
                                hour,
                                queue_size,
                                round(recent_avg_wait, 2),
                                round(wait_time, 2)
                            ])

                        print(
                            f"Logged -> Hour: {hour}  |  Queue: {queue_size}  |  "
                            f"Avg Wait: {recent_avg_wait:.1f}s  |  This Wait: {wait_time:.1f}s"
                        )

                    if inside_queue:
                        if track_id not in tracked:
                            tracked[track_id] = {
                                "enter_time": current_time,
                                "last_seen": current_time,
                                "counted": False,
                                "served": False,
                                "inside_cashier": False
                            }
                        tracked[track_id]["last_seen"] = current_time
                        tracked[track_id]["inside_cashier"] = inside_cashier

                        if current_time - tracked[track_id]["enter_time"] >= 3:
                            tracked[track_id]["counted"] = True

            # Remove people not seen for 2 video-seconds
            for person_id in list(tracked.keys()):
                if current_time - tracked[person_id]["last_seen"] > 2:
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
