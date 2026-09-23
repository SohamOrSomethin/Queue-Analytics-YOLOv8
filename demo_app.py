import os
import joblib
import gradio as gr
from ultralytics import YOLO

# Import the existing pipeline functions
from app import load_saved_roi_state
from process_video import process_video

# 1. Load the Models
print("Loading YOLOv8 and XGBoost models...")
yolo_model = YOLO("yolov8n.pt")

if os.path.exists("xgboost_model.pkl"):
    xgb_model = joblib.load("xgboost_model.pkl")
else:
    xgb_model = None
    print("Warning: xgboost_model.pkl not found!")

# 2. Main Inference Wrapper
def run_demo(video_input, yt_url):
    # Resolve the video path
    video_path = yt_url if yt_url else video_input
    
    if not video_path:
        yield None, "Please upload a video or enter a YouTube URL."
        return

    # Load the business logic (ROIs) you already drew!
    roi_state, loaded = load_saved_roi_state()
    if not loaded or not roi_state["queue_rois"] or not roi_state["cashier_rois"]:
        yield None, "Please run app.py first to draw and save your Queue/Cashier ROIs!"
        return

    import numpy as np
    queue_rois = [np.array([roi], dtype=np.int32) for roi in roi_state["queue_rois"]]
    cashier_rois = [np.array([roi], dtype=np.int32) for roi in roi_state["cashier_rois"]]

    is_youtube = "youtube.com" in video_path or "youtu.be" in video_path

    # Stream the annotated frames and text back to the UI
    for frame, status_text in process_video(
        video_path=video_path,
        model=yolo_model,
        queue_rois=queue_rois,
        cashier_rois=cashier_rois,
        frame_skip=2,
        show_window=False,
        gradio_mode=True,
        generate_dataset=False,
        is_youtube=is_youtube,
        xgb_model=xgb_model
    ):
        yield frame, status_text


# 3. Build the Premium UI
# We use a beautiful dark monochrome theme and custom CSS to look sleek and professional
theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
}
.box-rounded {
    border-radius: 15px !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
    border: 1px solid #3d3d5c !important;
    overflow: hidden;
}
button.primary {
    border-radius: 20px !important;
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(124, 58, 237, 0.4) !important;
}
"""

with gr.Blocks(css=custom_css, title="Queue Analytics AI") as demo:
    gr.Markdown(
        """
        # Queue & Wait-Time Analytics
        *End-to-end computer vision and predictive wait-time analytics.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["box-rounded"]):
            video_input = gr.Video(label="Upload CCTV Stream (Optional)")
            yt_url = gr.Textbox(label="YouTube Live URL (Optional)", placeholder="https://www.youtube.com/watch?v=...")
            start_btn = gr.Button("Analyze Queue", variant="primary")
            
        with gr.Column(scale=2, elem_classes=["box-rounded"]):
            output_video = gr.Image(label="Live YOLOv8 Tracking", interactive=False)
            wait_time_text = gr.Textbox(label="Live Wait Time Metrics", lines=6, interactive=False)
            
    # Connect the button
    start_btn.click(
        fn=run_demo, 
        inputs=[video_input, yt_url], 
        outputs=[output_video, wait_time_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=theme)
