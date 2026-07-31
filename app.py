import gradio as gr
from ultralytics import YOLO

from process_video import load_rois, stream_video_processing

yolo_model = YOLO("yolov8n.pt")
queue_roi, cashier_rois = load_rois("rois.json")


def resolve_video_path(video_input):
    if video_input is None:
        return "video2.mp4"

    if isinstance(video_input, str):
        return video_input

    if isinstance(video_input, dict):
        if "path" in video_input and video_input["path"]:
            return video_input["path"]
        if "video" in video_input and video_input["video"]:
            return video_input["video"]

    return "video2.mp4"


def run_live_monitor(video_input):
    video_path = resolve_video_path(video_input)

    for frame, status_text in stream_video_processing(
        video_path=video_path,
        model=yolo_model,
        queue_roi=queue_roi,
        cashier_rois=cashier_rois,
        frame_skip=2
    ):
        yield frame, status_text


with gr.Blocks() as demo:
    gr.Markdown("# DineFlow AI: Live Queue & Wait-Time Monitor")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload CCTV Video")
            submit_btn = gr.Button("Start Monitoring", variant="primary")

        with gr.Column():
            live_frame = gr.Image(label="Live Processed Frame", type="numpy")
            result_output = gr.Textbox(label="Live Wait Time Result", lines=5)

    submit_btn.click(
        fn=run_live_monitor,
        inputs=[video_input],
        outputs=[live_frame, result_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())