import cv2
import numpy as np
from datetime import datetime
import json

def load_rois(json_path="rois.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    queue_rois = [np.array([roi], dtype=np.int32) for roi in data["queue_rois"]]
    cashier_rois = [np.array([roi], dtype=np.int32) for roi in data["cashier_rois"]]
    return queue_rois, cashier_rois

def process_video(video_path, model, queue_rois, cashier_rois, show_window=True, frame_skip=2):
    cap = cv2.VideoCapture(video_path)

    # FIX 1: Use video FPS to calculate video-based timestamps instead of wall-clock time.
    # This means frame_skip, slow inference, and CPU lag no longer distort wait times.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    recent_waits = []
    tracked = {}
    count = 0
    queue_counts = [0] * len(queue_rois)
    hour = datetime.now().hour
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_index += 1

        if frame_index % frame_skip != 0:
            continue

        # FIX 1: current_time is now video time in seconds, not wall-clock time.
        current_time = frame_index / fps

        results = model.track(
            frame,
            persist=True,  # remember people from the previous frame and assign them an ID
            verbose=False
        )
        count = 0
        queue_counts = [0] * len(queue_rois)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])

                if cls == 0:  # class 0 means a person
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
                        print(f"Person {track_id} served after {wait_time:.2f} seconds (video time)")
                        recent_waits.append((track_id, wait_time))
                        print("Recent waits:")
                        for i, (pid, t) in enumerate(recent_waits, start=1):
                            print(f"{i}: Person {pid} -> {t:.2f}s")
                        print("-" * 40)

                        tracked[track_id]["served"] = True
                        if len(recent_waits) > 20:
                            recent_waits.pop(0)

                    if inside_queue_id is not None:
                        if track_id not in tracked:
                            tracked[track_id] = {
                                "enter_time": current_time,
                                "last_seen": current_time,
                                "counted": False,
                                "served": False,
                                "queue_id": inside_queue_id,
                                "inside_cashier": False  # FIX 2: track cashier state per person
                            }

                        tracked[track_id]["last_seen"] = current_time
                        tracked[track_id]["queue_id"] = inside_queue_id
                        # FIX 2: update cashier state every frame
                        tracked[track_id]["inside_cashier"] = inside_cashier

                        time_inside = current_time - tracked[track_id]["enter_time"]
                        if time_inside >= 3 and not tracked[track_id]["counted"]:
                            tracked[track_id]["counted"] = True

        for person_id in list(tracked.keys()):
            if current_time - tracked[person_id]["last_seen"] > 2:
                del tracked[person_id]

        for info in tracked.values():
            # FIX 2: exclude people at the cashier from the queue count.
            # Previously the cashier ROI overlaps the queue ROI, so cashier
            # staff/customers were being double-counted in the queue.
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

        hour = datetime.now().hour

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

        if show_window:
            cv2.imshow("People Detection", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()

    if show_window:
        cv2.destroyAllWindows()

    if len(recent_waits) > 0:
        recent_avg_wait = sum(t for _, t in recent_waits) / len(recent_waits)
    else:
        recent_avg_wait = 0

    return {
        "queue_count": count,
        "queue_counts": queue_counts,
        "recent_waits": recent_waits,
        "recent_avg_wait": recent_avg_wait,
        "hour": hour
    }


def stream_video_processing(video_path, model, queue_rois, cashier_rois, frame_skip=2):
    cap = cv2.VideoCapture(video_path)

    # FIX 1: Use video FPS so wait times reflect actual video duration, not wall-clock time.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    recent_waits = []
    tracked = {}
    count = 0
    queue_counts = [0] * len(queue_rois)
    frame_index = 0
    hour = datetime.now().hour
    recent_avg_wait = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_index += 1

        if frame_index % frame_skip != 0:
            continue

        # FIX 1: video-time timestamp, not wall clock.
        current_time = frame_index / fps

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

                if cls != 0:
                    continue

                if box.id is None:
                    continue

                conf = float(box.conf[0])
                name = model.names[cls]
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
                    recent_waits.append((track_id, wait_time))
                    tracked[track_id]["served"] = True

                    if len(recent_waits) > 20:
                        recent_waits.pop(0)

                if inside_queue_id is not None:
                    if track_id not in tracked:
                        tracked[track_id] = {
                            "enter_time": current_time,
                            "last_seen": current_time,
                            "counted": False,
                            "served": False,
                            "queue_id": inside_queue_id,
                            "inside_cashier": False  # FIX 2: track cashier state per person
                        }

                    tracked[track_id]["last_seen"] = current_time
                    tracked[track_id]["queue_id"] = inside_queue_id
                    # FIX 2: update cashier state every frame
                    tracked[track_id]["inside_cashier"] = inside_cashier

                    time_inside = current_time - tracked[track_id]["enter_time"]
                    if time_inside >= 3 and not tracked[track_id]["counted"]:
                        tracked[track_id]["counted"] = True

        for person_id in list(tracked.keys()):
            if current_time - tracked[person_id]["last_seen"] > 2:
                del tracked[person_id]

        for info in tracked.values():
            # FIX 2: do not count people who are currently at the cashier in the queue total.
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

        hour = datetime.now().hour

        cv2.putText(frame, f"Queue Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Avg Wait: {recent_avg_wait:.1f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        queue_lines_text = "\n".join(
            [f"Queue {i + 1}: {queue_counts[i]} people" for i in range(len(queue_counts))]
        )

        status_text = (
            f"Detected Queue Size: {count} people\n"
            f"{queue_lines_text}\n"
            f"Hour: {hour}:00\n"
            f"Average Wait Time: {recent_avg_wait:.2f} seconds\n"
            f"Served Count Sample Size: {len(recent_waits)}"
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame_rgb, status_text

    cap.release()

    final_queue_lines_text = "\n".join(
        [f"Queue {i + 1}: {queue_counts[i]} people" for i in range(len(queue_counts))]
    )

    final_text = (
        f"Detected Queue Size: {count} people\n"
        f"{final_queue_lines_text}\n"
        f"Hour: {hour}:00\n"
        f"Average Wait Time: {recent_avg_wait:.2f} seconds\n"
        f"Processing complete."
    )
    yield None, final_text