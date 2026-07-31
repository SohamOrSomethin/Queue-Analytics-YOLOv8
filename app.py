import gradio as gr
from ultralytics import YOLO
import yt_dlp

from process_video import load_rois, stream_video_processing

yolo_model = YOLO("yolov8n.pt")
queue_roi, cashier_rois = load_rois("rois.json")


def resolve_video_path(video_input, yt_url=None):
    if yt_url and ("youtube.com" in yt_url or "youtu.be" in yt_url):
        # Use yt-dlp to extract stream URL
        ydl_opts = {'format': 'best'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(yt_url, download=False)
            return info_dict.get('url', "video2.mp4")

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


def run_live_monitor(video_input, yt_url):
    video_path = resolve_video_path(video_input, yt_url)

    for frame, status_text in stream_video_processing(
        video_path=video_path,
        model=yolo_model,
        queue_roi=queue_roi,
        cashier_rois=cashier_rois,
        frame_skip=2
    ):
        yield frame, status_text


with gr.Blocks() as demo:
    gr.Markdown("# Queue Analytics: Live Queue & Wait-Time Monitor")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload CCTV Video (Optional)")
            yt_url = gr.Textbox(label="YouTube Live URL", placeholder="https://www.youtube.com/watch?v=...", value="https://www.youtube.com/watch?v=CmtuOVxcKRo")
            submit_btn = gr.Button("Start Monitoring", variant="primary")

        with gr.Column():
            live_frame = gr.Image(label="Live Processed Frame", type="numpy")
            result_output = gr.Textbox(label="Live Wait Time Result", lines=5)

    submit_btn.click(
        fn=run_live_monitor,
        inputs=[video_input, yt_url],
        outputs=[live_frame, result_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())