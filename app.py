import gradio as gr
import cv2
import numpy as np
import pandas as pd
import joblib
import time
from ultralytics import YOLO

# 1. Load Models
xgb_model = joblib.load('xgboost_model.pkl')
yolo_model = YOLO("yolov8n.pt")

# The Region of Interest from dryrun.py
queue_roi = np.array([
    [318, 367], [181, 152], [174, 107], [391,  19],
    [440, 116], [381, 150], [340, 176], [434, 302],
    [451, 330]
], dtype=np.int32)

def process_video_and_predict(video_path, hour, party_size, recent_avg_wait_time):
    if not video_path:
        return "Please upload a video.", None

    cap = cv2.VideoCapture(video_path)
    
    # We will save the processed video to show in Gradio
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    out_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracked = {}
    final_count = 0
    frame_count = 0
    max_frames_to_process = fps * 5  # Process 5 seconds of video to keep it fast

    while True:
        success, frame = cap.read()
        if not success or frame_count > max_frames_to_process:
            break

        results = yolo_model.track(frame, persist=True, verbose=False)
        video_time = frame_count / fps  # Use video time instead of wall-clock time
        
        # Draw ROI
        cv2.polylines(frame, [queue_roi], True, (255, 0, 0), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Draw person tracking
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                    # Check if inside ROI
                    inside = cv2.pointPolygonTest(queue_roi, (center_x, center_y), False)
                    if inside >= 0:
                        if track_id not in tracked:
                            tracked[track_id] = {"enter_time": video_time, "last_seen": video_time, "counted": False}
                        
                        tracked[track_id]["last_seen"] = video_time
                        time_inside = video_time - tracked[track_id]["enter_time"]
                        
                        if time_inside >= 1.0 and not tracked[track_id]["counted"]:  
                            tracked[track_id]["counted"] = True

        # Cleanup lost tracks
        for person_id in list(tracked.keys()):
            if video_time - tracked[person_id]["last_seen"] > 2.0:
                del tracked[person_id]

        final_count = sum(info["counted"] for info in tracked.values())
        
        cv2.putText(frame, f"Queue Count: {final_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # STEP B: PREDICT WAIT TIME USING XGBOOST
    features = pd.DataFrame([[hour, party_size, final_count, recent_avg_wait_time]], 
                            columns=['hour', 'party_size', 'queue_size', 'recent_avg_wait_time'])
    
    ewt = xgb_model.predict(features)[0]
    
    result_text = (
        f"👥 Detected Queue Size: {final_count} people\n"
        f"⏱️ Estimated Wait Time: {ewt:.1f} minutes"
    )
    
    return result_text, out_path

# 3. GRADIO UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕒 DineFlow AI: Queue & Wait-Time Predictor")
    
    with gr.Row():
        with gr.Column():
            hour_input = gr.Number(label="Hour of Day (9-22)", value=18)
            party_input = gr.Number(label="Party Size", value=2)
            recent_wait_input = gr.Number(label="Recent Avg Wait (mins)", value=15.0)
            video_input = gr.Video(label="Upload CCTV Video")
            submit_btn = gr.Button("Calculate Wait Time", variant="primary")
            
        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result", lines=2)
            video_output = gr.Video(label="Processed Video Stream")
            
    submit_btn.click(
        fn=process_video_and_predict,
        inputs=[video_input, hour_input, party_input, recent_wait_input],
        outputs=[result_output, video_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)