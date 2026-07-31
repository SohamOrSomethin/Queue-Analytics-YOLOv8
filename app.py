import json
import cv2
import numpy as np
import gradio as gr
import yt_dlp
from ultralytics import YOLO

from process_video import stream_video_processing

yolo_model = YOLO("yolov8n.pt")


def resolve_video_path(video_input, yt_url=None):
    if yt_url and ("youtube.com" in yt_url or "youtu.be" in yt_url):
        # Use yt-dlp to extract stream URL
        ydl_opts = {"format": "best"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(yt_url, download=False)
            return info_dict.get("url", "video2.mp4")

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

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        empty_state = {"queue_rois": [], "cashier_rois": [], "current_points": []}
        return None, None, empty_state, "Could not read the first frame from the selected video."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi_state = {
        "queue_rois": [],
        "cashier_rois": [],
        "current_points": []
    }

    return video_path, frame_rgb, roi_state, "First frame loaded. Click points to start drawing an ROI."


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


def run_live_monitor(video_path, roi_state):
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

    for frame, status_text in stream_video_processing(
        video_path=video_path,
        model=yolo_model,
        queue_rois=queue_rois,
        cashier_rois=cashier_rois,
        frame_skip=2
    ):
        yield frame, status_text


with gr.Blocks() as demo:
    gr.Markdown("# Queue Analytics: ROI Setup + Live Queue Monitor")

    video_path_state = gr.State()
    base_frame_state = gr.State()
    roi_state = gr.State({
        "queue_rois": [],
        "cashier_rois": [],
        "current_points": []
    })

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload CCTV Video (Optional)")
            yt_url = gr.Textbox(
                label="YouTube Live URL (Optional)",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            load_frame_btn = gr.Button("Load First Frame", variant="primary")

            roi_type = gr.Radio(
                choices=["Queue", "Cashier"],
                value="Queue",
                label="ROI Type"
            )

            save_polygon_btn = gr.Button("Save Current Polygon")
            undo_btn = gr.Button("Undo Last Point")
            clear_btn = gr.Button("Clear Current Polygon")
            save_rois_btn = gr.Button("Save All ROIs to rois.json")
            start_btn = gr.Button("Start Monitoring", variant="primary")

        with gr.Column():
            roi_image = gr.Image(
                label="First Frame ROI Editor",
                type="numpy",
                interactive=True
            )
            roi_status = gr.Textbox(label="ROI Status", lines=6)
            roi_summary = gr.Textbox(label="ROI Summary", lines=4)
            live_frame = gr.Image(label="Live Processed Frame", type="numpy")
            result_output = gr.Textbox(label="Live Wait Time Result", lines=10)

    load_frame_btn.click(
        fn=extract_first_frame,
        inputs=[video_input, yt_url],
        outputs=[video_path_state, roi_image, roi_state, roi_status]
    ).then(
        fn=lambda img: img,
        inputs=[roi_image],
        outputs=[base_frame_state]
    ).then(
        fn=format_roi_summary,
        inputs=[roi_state],
        outputs=[roi_summary]
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

    start_btn.click(
        fn=run_live_monitor,
        inputs=[video_path_state, roi_state],
        outputs=[live_frame, result_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())