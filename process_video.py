import cv2
import numpy as np
from datetime import datetime
import json
import csv
import yt_dlp
import os
import subprocess
import time
import imageio_ffmpeg

try:
    from predict_wait import format_wait_time
except Exception:
    def format_wait_time(seconds):
        seconds = max(0, int(round(seconds)))
        return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"

MIN_VALID_WAIT = 1
MAX_VALID_WAIT = 200
HISTORIC_CSV_CANDIDATES = ["queue_data_cleaned.csv", "queue_data.csv"]
MAX_RECONNECTS = 5


def load_rois(json_path="rois.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    queue_rois = [np.array([roi], dtype=np.int32) for roi in data["queue_rois"]]
    cashier_rois = [np.array([roi], dtype=np.int32) for roi in data["cashier_rois"]]
    return queue_rois, cashier_rois


def get_historic_mean_wait(csv_candidates=HISTORIC_CSV_CANDIDATES):
    for path in csv_candidates:
        if not os.path.exists(path):
            continue
        try:
            waits = []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        w = float(row["actual_wait"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if MIN_VALID_WAIT <= w <= MAX_VALID_WAIT:
                        waits.append(w)
            if waits:
                return sum(waits) / len(waits)
        except Exception:
            continue
    return None


def process_video(
    video_path,
    model,
    queue_rois,
    cashier_rois,
    show_window=True,
    generate_dataset=False,
    csv_path=None,
    frame_skip=2,
    is_youtube=False,
    gradio_mode=False,
    predictor=None
):

    csv_file = None
    writer = None

    if generate_dataset:
        file_exists = os.path.exists(csv_path)
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "hour",
                "queue_size",
                "recent_avg_wait_time",
                "actual_wait"
            ])
            csv_file.flush()
            os.fsync(csv_file.fileno())

    pipe = None
    cap = None

    if is_youtube:
        stream_url, fps, width, height = resolve_youtube_stream(video_path)
        pipe = open_ffmpeg_pipe(stream_url, width, height)
        frame_size = width * height * 3
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    historic_mean = get_historic_mean_wait()

    recent_waits = []
    tracked = {}
    count = 0
    queue_counts = [0] * len(queue_rois)
    frame_index = 0
    recent_avg_wait = 0
    hour = datetime.now().hour
    reconnects = 0

    try:
        while True:
            new_wait = None
            event_timestamp = None
            event_hour = None

            if is_youtube:
                raw = pipe.stdout.read(frame_size)
                if len(raw) < frame_size:
                    reconnects += 1
                    if reconnects > MAX_RECONNECTS:
                        print("Stream ended and reconnect limit reached. Stopping.")
                        break
                    print(f"Stream ended early, reconnecting ({reconnects}/{MAX_RECONNECTS})...")
                    pipe.kill()
                    time.sleep(2)
                    try:
                        stream_url, fps, width, height = resolve_youtube_stream(video_path)
                        pipe = open_ffmpeg_pipe(stream_url, width, height)
                        frame_size = width * height * 3
                    except Exception as e:
                        print(f"Reconnect failed: {e}")
                        break
                    continue
                reconnects = 0
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                frame = frame.copy()
            else:
                success, frame = cap.read()
                if not success:
                    break

            frame_index += 1

            if frame_index % frame_skip != 0:
                continue

            current_time = frame_index / fps
            hour = datetime.now().hour

            results = model.track(
                frame,
                persist=True,
                verbose=False
            )
            count = 0
            queue_counts = [0] * len(queue_rois)

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])

                    if cls == 0:
                        conf = float(box.conf[0])
                        name = model.names[cls]

                        if box.id is None:
                            continue

                        track_id = int(box.id[0])
                        label = f"ID:{track_id} {name} {conf:.2f}"

                        x1, y1, x2, y2 = box.xyxy[0]

                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2)

                        cv2.putText(
                            frame,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                        inside_cashier = False
                        for roi in cashier_rois:
                            if cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0:
                                inside_cashier = True
                                break

                        inside_queue_id = None
                        for idx, roi in enumerate(queue_rois):
                            inside = cv2.pointPolygonTest(roi, (center_x, center_y), False)
                            if inside >= 0:
                                inside_queue_id = idx
                                break

                        if (
                            inside_cashier
                            and track_id in tracked
                            and tracked[track_id]["counted"]
                            and not tracked[track_id]["served"]
                        ):
                            wait_time = current_time - tracked[track_id]["enter_time"]
                            tracked[track_id]["served"] = True
                            if wait_time <= 3.5:
                                continue
                            new_wait = wait_time
                            print(f"Person {track_id} served after {wait_time:.2f} seconds (video time)")
                            recent_waits.append((track_id, wait_time))

                            if len(recent_waits) > 20:
                                recent_waits.pop(0)

                            now = datetime.now()
                            event_hour = now.hour
                            event_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                        if inside_queue_id is not None:
                            if track_id not in tracked:
                                tracked[track_id] = {
                                    "enter_time": current_time,
                                    "last_seen": current_time,
                                    "counted": False,
                                    "served": False,
                                    "queue_id": inside_queue_id,
                                    "inside_cashier": False
                                }

                            tracked[track_id]["last_seen"] = current_time
                            tracked[track_id]["queue_id"] = inside_queue_id
                            tracked[track_id]["inside_cashier"] = inside_cashier

                            time_inside = current_time - tracked[track_id]["enter_time"]
                            if time_inside >= 3 and not tracked[track_id]["counted"]:
                                tracked[track_id]["counted"] = True

            for person_id in list(tracked.keys()):
                if current_time - tracked[person_id]["last_seen"] > 2:
                    del tracked[person_id]

            for info in tracked.values():
                if (
                    info["counted"]
                    and info.get("queue_id") is not None
                    and not info.get("inside_cashier", False)
                ):
                    queue_counts[info["queue_id"]] += 1

            count = sum(queue_counts)

            for idx, roi in enumerate(queue_rois):
                cv2.polylines(frame, [roi], True, (255, 0, 0), 2)
                first_point = roi[0][0]
                cv2.putText(
                    frame,
                    f"Queue {idx + 1}: {queue_counts[idx]}",
                    (int(first_point[0]), int(first_point[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            for roi in cashier_rois:
                cv2.polylines(frame, [roi], True, (0, 255, 255), 2)

            if len(recent_waits) > 0:
                recent_avg_wait = sum(t for _, t in recent_waits) / len(recent_waits)
            else:
                recent_avg_wait = 0

            if generate_dataset and new_wait is not None:
                writer.writerow([
                    event_timestamp,
                    event_hour,
                    count,
                    round(recent_avg_wait, 2),
                    round(new_wait, 2)
                ])
                csv_file.flush()
                os.fsync(csv_file.fileno())

            if len(recent_waits) > 0:
                avg_for_model = recent_avg_wait
                basis = "live average"
            elif historic_mean is not None:
                avg_for_model = historic_mean
                basis = "historic mean (seed)"
            else:
                avg_for_model = None
                basis = "none"

            expected_wait_text = "Collecting data..."
            if predictor is None:
                expected_wait_text = "Model not loaded"
            elif count <= 0:
                expected_wait_text = "No one in queue"
            elif avg_for_model is not None:
                try:
                    predicted = predictor.predict(hour, count, avg_for_model)
                    expected_wait_text = format_wait_time(predicted)
                except Exception as e:
                    expected_wait_text = f"Prediction error: {e}"

            cv2.putText(
                frame,
                f"Queue Count: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Avg Wait: {recent_avg_wait:.1f}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Expected Wait: {expected_wait_text}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if gradio_mode:
                queue_lines_text = "\n".join(
                    [f"Queue {i+1}: {queue_counts[i]} people" for i in range(len(queue_counts))]
                )

                status_text = (
                    f"Detected Queue Size: {count} people\n"
                    f"{queue_lines_text}\n"
                    f"Hour: {hour}:00\n"
                    f"Average Wait Time: {recent_avg_wait:.2f} seconds\n"
                    f"Expected Wait Time: {expected_wait_text}\n"
                    f"Prediction Basis: {basis}\n"
                    f"Served Count Sample Size: {len(recent_waits)}"
                )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb, status_text

            elif show_window:
                cv2.imshow("People Detection", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

    finally:
        if pipe is not None:
            pipe.kill()
        if cap is not None:
            cap.release()
        if show_window:
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    if not gradio_mode:
        return {
            "queue_count": count,
            "queue_counts": queue_counts,
            "recent_waits": recent_waits,
            "recent_avg_wait": recent_avg_wait,
            "hour": hour
        }


def open_ffmpeg_pipe(stream_url, width, height):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        '-loglevel', 'warning',
        '-reconnect', '1',
        '-reconnect_streamed', '1',
        '-reconnect_delay_max', '5',
        '-i', stream_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vf', f'scale={width}:{height}',
        'pipe:1'
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)


def resolve_youtube_stream(yt_url):
    """
    Use yt-dlp to extract stream metadata.
    Returns (stream_url, fps, width, height).
    """
    selectors = [
        "b[height<=720]",
        "bv*[height<=720]",
        "b/bv*",
    ]

    last_error = None
    for fmt in selectors:
        ydl_opts = {
            "format": fmt,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(yt_url, download=False)

            stream_url = info.get("url")
            if not stream_url and info.get("requested_formats"):
                stream_url = info["requested_formats"][0].get("url")

            if not stream_url:
                continue

            fps = float(info.get("fps") or 30)
            width = int(info.get("width") or 1280)
            height = int(info.get("height") or 720)
            return stream_url, fps, width, height
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"No usable stream format found: {last_error}")