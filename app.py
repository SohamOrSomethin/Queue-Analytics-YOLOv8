import json
import cv2
import numpy as np
import gradio as gr
import yt_dlp
from ultralytics import YOLO

from process_video import process_video
from process_video import resolve_youtube_stream
from process_video import open_ffmpeg_pipe

yolo_model = YOLO("yolov8n.pt")


def load_saved_roi_state():
    """
    Load queue_rois / cashier_rois from rois.json on disk, if it exists,
    into the same dict shape used by roi_state. Returns an empty state
    if the file doesn't exist or can't be parsed.
    """
    import os
    empty_state = {"queue_rois": [], "cashier_rois": [], "current_points": []}
    if not os.path.exists("rois.json"):
        return empty_state, False

    try:
        with open("rois.json", "r") as f:
            data = json.load(f)
        loaded_state = {
            "queue_rois": data.get("queue_rois", []),
            "cashier_rois": data.get("cashier_rois", []),
            "current_points": []
        }
        return loaded_state, True
    except Exception:
        return empty_state, False


def resolve_video_path(video_input, yt_url=None):
    if yt_url:
        return yt_url

    if video_input is None:
        return None

    if isinstance(video_input, str):
        return video_input

    if isinstance(video_input, dict):
        if "path" in video_input and video_input["path"]:
            return video_input["path"]
        if "video" in video_input and video_input["video"]:
            return video_input["video"]

    return None


def extract_first_frame(video_input, yt_url):
    video_path = resolve_video_path(video_input, yt_url)

    if not video_path:
        empty_state = {"queue_rois": [], "cashier_rois": [], "current_points": []}
        return None, None, empty_state, "Please upload a video or enter a YouTube live URL."

    is_youtube = (
        "youtube.com" in video_path
        or
        "youtu.be" in video_path
    )

    if is_youtube:
        try:
            stream_url, fps, width, height = resolve_youtube_stream(video_path)
        except Exception as e:
            empty_state = {"queue_rois": [], "cashier_rois": [], "current_points": []}
            return None, None, empty_state, f"Could not resolve YouTube stream: {e}"

        pipe = open_ffmpeg_pipe(stream_url, width, height)
        frame_size = width * height * 3
        raw = pipe.stdout.read(frame_size)
        pipe.kill()

        if len(raw) < frame_size:
            empty_state = {"queue_rois": [], "cashier_rois": [], "current_points": []}
            return None, None, empty_state, "Could not read the first frame from the YouTube stream."

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
    else:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            empty_state = {"queue_rois": [], "cashier_rois": [], "current_points": []}
            return None, None, empty_state, "Could not read the first frame from the selected video."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi_state, loaded_from_file = load_saved_roi_state()

    if loaded_from_file:
        n_queue = len(roi_state["queue_rois"])
        n_cashier = len(roi_state["cashier_rois"])
        message = (
            f"First frame loaded. Loaded saved ROIs from rois.json "
            f"({n_queue} queue, {n_cashier} cashier). "
            f"Click points to draw a new ROI, or start generating dataset directly."
        )
    else:
        message = "First frame loaded. Click points to start drawing an ROI."

    # Return the RAW frame here — the overlay (if any saved ROIs were
    # loaded) gets drawn in a separate .then() step in the click handler,
    # so base_frame_state stays a clean, overlay-free copy for future
    # point-adding/undo redraws.
    return video_path, frame_rgb, roi_state, message


def draw_roi_overlay(base_img, roi_state):
    if base_img is None:
        return None

    img = base_img.copy()

    for i, roi in enumerate(roi_state["queue_rois"]):
        pts = np.array(roi, dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 0), 2)
        x, y = pts[0]
        cv2.putText(
            img,
            f"Queue {i + 1}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    for i, roi in enumerate(roi_state["cashier_rois"]):
        pts = np.array(roi, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        x, y = pts[0]
        cv2.putText(
            img,
            f"Cashier {i + 1}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    for point in roi_state["current_points"]:
        x, y = point
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1)

    if len(roi_state["current_points"]) > 1:
        pts = np.array(roi_state["current_points"], dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 0, 255), 2)

    return img


def add_point(base_img, roi_state, evt: gr.SelectData):
    if base_img is None:
        return None, roi_state, "Load a video first."

    if evt.index is None:
        updated = draw_roi_overlay(base_img, roi_state)
        return updated, roi_state, "Could not read click coordinates."

    x = int(evt.index[0])
    y = int(evt.index[1])
    roi_state["current_points"].append([x, y])

    updated = draw_roi_overlay(base_img, roi_state)
    return updated, roi_state, f"Added point ({x}, {y}). Current polygon has {len(roi_state['current_points'])} points."


def undo_last_point(base_img, roi_state):
    if roi_state["current_points"]:
        roi_state["current_points"].pop()

    updated = draw_roi_overlay(base_img, roi_state)
    return updated, roi_state, f"Current polygon has {len(roi_state['current_points'])} points."


def clear_current_polygon(base_img, roi_state):
    roi_state["current_points"] = []
    updated = draw_roi_overlay(base_img, roi_state)
    return updated, roi_state, "Cleared current unsaved polygon."


def save_current_polygon(base_img, roi_state, roi_type):
    if len(roi_state["current_points"]) < 3:
        updated = draw_roi_overlay(base_img, roi_state)
        return updated, roi_state, "Need at least 3 points to save a polygon."

    polygon = roi_state["current_points"][:]

    if roi_type == "Queue":
        roi_state["queue_rois"].append(polygon)
        message = f"Saved Queue ROI {len(roi_state['queue_rois'])}."
    else:
        roi_state["cashier_rois"].append(polygon)
        message = f"Saved Cashier ROI {len(roi_state['cashier_rois'])}."

    roi_state["current_points"] = []
    updated = draw_roi_overlay(base_img, roi_state)
    return updated, roi_state, message


def load_saved_rois_onto_frame(base_img):
    roi_state, loaded_from_file = load_saved_roi_state()

    if base_img is None:
        return None, roi_state, "Load a video frame first before loading saved ROIs."

    if not loaded_from_file:
        return base_img, roi_state, "No rois.json found on disk yet."

    updated = draw_roi_overlay(base_img, roi_state)
    n_queue = len(roi_state["queue_rois"])
    n_cashier = len(roi_state["cashier_rois"])
    return updated, roi_state, f"Loaded saved ROIs from rois.json ({n_queue} queue, {n_cashier} cashier)."


def save_rois_to_file(roi_state, output_path="rois.json"):
    data = {
        "queue_rois": roi_state["queue_rois"],
        "cashier_rois": roi_state["cashier_rois"]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return f"Saved ROIs to {output_path}"


def format_roi_summary(roi_state):
    return (
        f"Queue ROIs: {len(roi_state['queue_rois'])}\n"
        f"Cashier ROIs: {len(roi_state['cashier_rois'])}\n"
        f"Current unsaved points: {len(roi_state['current_points'])}"
    )


def run_live_monitor(video_path, roi_state, frame_skip):
    if not video_path:
        yield None, "Please load a video first."
        return

    queue_rois = [
        np.array([roi], dtype=np.int32)
        for roi in roi_state["queue_rois"]
    ]
    cashier_rois = [
        np.array([roi], dtype=np.int32)
        for roi in roi_state["cashier_rois"]
    ]

    if not queue_rois:
        yield None, "Please mark at least one Queue ROI before starting."
        return

    if not cashier_rois:
        yield None, "Please mark at least one Cashier ROI before starting."
        return

    is_youtube = (
        "youtube.com" in video_path
        or
        "youtu.be" in video_path
        )
    
    for frame,status in process_video(
        video_path,
        model=yolo_model,
        queue_rois=queue_rois,
        cashier_rois=cashier_rois,
        frame_skip=int(frame_skip),
        show_window=False,
        gradio_mode=True,
        generate_dataset=False,
        is_youtube=is_youtube
        ):
        yield frame,status

def generate_dataset(video_path, roi_state):
    """
    NOTE: this is a generator itself, so Gradio streams progress into the
    roi_status textbox live instead of the button appearing to finish
    instantly. process_video() contains `yield` statements, so simply
    calling it without iterating it does NOTHING.
    """
    import os
    import csv as _csv

    if not video_path:
        yield "Please load a video first."
        return

    queue_rois = [
        np.array([roi], dtype=np.int32)
        for roi in roi_state["queue_rois"]
    ]

    cashier_rois = [
        np.array([roi], dtype=np.int32)
        for roi in roi_state["cashier_rois"]
    ]
    is_youtube = (
        "youtube.com" in video_path
        or
        "youtu.be" in video_path
    )

    csv_path = "queue_data.csv"

    def count_rows():
        if not os.path.exists(csv_path):
            return 0
        with open(csv_path, newline="") as f:
            return max(0, sum(1 for _ in _csv.reader(f)) - 1)

    rows_before = count_rows()
    yield "Starting dataset generation... (resolving stream, this can take a few seconds)"

    frame_count = 0
    try:
        for frame, status in process_video(
            video_path=video_path,
            model=yolo_model,
            queue_rois=queue_rois,
            cashier_rois=cashier_rois,
            show_window=False,
            generate_dataset=True,
            csv_path=csv_path,
            is_youtube=is_youtube,
            gradio_mode=True,
        ):
            frame_count += 1
            if frame_count % 5 == 0:
                rows_now = count_rows() - rows_before
                yield f"Processing frame {frame_count}... (rows written so far: {rows_now})\n{status}"
    except Exception as e:
        yield f"Dataset generation stopped with an error: {e}"
        return

    rows_written = count_rows() - rows_before
    if rows_written <= 0:
        yield "Dataset generation complete. No valid events detected. Rows written: 0"
    else:
        yield f"Dataset generation complete. Rows written: {rows_written}"

with gr.Blocks() as demo:
    gr.Markdown("# Queue Analytics: ROI Setup + Live Queue Monitor")

    video_path_state = gr.State()
    base_frame_state = gr.State()
    roi_state = gr.State({
        "queue_rois": [],
        "cashier_rois": [],
        "current_points": []
    })

    with gr.Tabs():
        with gr.Tab("Step 1: Setup ROIs"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Load Video")
                    video_input = gr.Video(label="Upload CCTV Video (Optional)")
                    yt_url = gr.Textbox(
                        label="YouTube Live URL (Optional)",
                        placeholder="https://www.youtube.com/watch?v=..."
                    )
                    load_frame_btn = gr.Button("Load First Frame", variant="primary")

                    gr.Markdown("### 2. Draw ROIs")
                    roi_type = gr.Radio(
                        choices=["Queue", "Cashier"],
                        value="Queue",
                        label="ROI Type"
                    )
                    with gr.Row():
                        undo_btn = gr.Button("Undo Last Point")
                        clear_btn = gr.Button("Clear Current Polygon")
                    save_polygon_btn = gr.Button("Save Current Polygon", variant="secondary")
                    
                    gr.Markdown("### 3. Save Configuration")
                    save_rois_btn = gr.Button("Save All ROIs to rois.json", variant="primary")
                    load_rois_btn = gr.Button("Load Saved ROIs", variant="secondary")
                    gen_data_btn = gr.Button("Start generating dataset", variant="primary")
                    
                    roi_status = gr.Textbox(label="ROI Status", lines=3)
                    roi_summary = gr.Textbox(label="ROI Summary", lines=4)

                with gr.Column(scale=2):
                    roi_image = gr.Image(
                        label="First Frame ROI Editor",
                        type="numpy",
                        interactive=True
                    )

        with gr.Tab("Step 2: Live Monitor"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Start Analytics")
                    frame_skip_slider = gr.Slider(minimum=1, maximum=30, value=2, step=1, label="Frame Skip (Higher = Faster, Lower = More Accurate)")
                    start_btn = gr.Button("Start Monitoring", variant="primary")
                    result_output = gr.Textbox(label="Live Wait Time Result", lines=10)
                
                with gr.Column(scale=2):
                    live_frame = gr.Image(label="Live Processed Frame", type="numpy")


    load_frame_btn.click(
        fn=extract_first_frame,
        inputs=[video_input, yt_url],
        outputs=[video_path_state, roi_image, roi_state, roi_status]
    ).then(
        fn=lambda img: img,
        inputs=[roi_image],
        outputs=[base_frame_state]
    ).then(
        fn=draw_roi_overlay,
        inputs=[base_frame_state, roi_state],
        outputs=[roi_image]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    gen_data_btn.click(
    fn=generate_dataset,
    inputs=[
        video_path_state,
        roi_state
    ],
    outputs=roi_status
    )



    roi_image.select(
        fn=add_point,
        inputs=[base_frame_state, roi_state],
        outputs=[roi_image, roi_state, roi_status]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    undo_btn.click(
        fn=undo_last_point,
        inputs=[base_frame_state, roi_state],
        outputs=[roi_image, roi_state, roi_status]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    clear_btn.click(
        fn=clear_current_polygon,
        inputs=[base_frame_state, roi_state],
        outputs=[roi_image, roi_state, roi_status]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    save_polygon_btn.click(
        fn=save_current_polygon,
        inputs=[base_frame_state, roi_state, roi_type],
        outputs=[roi_image, roi_state, roi_status]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    save_rois_btn.click(
        fn=save_rois_to_file,
        inputs=[roi_state],
        outputs=[roi_status]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    load_rois_btn.click(
        fn=load_saved_rois_onto_frame,
        inputs=[base_frame_state],
        outputs=[roi_image, roi_state, roi_status]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
    )

    start_btn.click(
        fn=run_live_monitor,
        inputs=[video_path_state, roi_state, frame_skip_slider],
        outputs=[live_frame, result_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())