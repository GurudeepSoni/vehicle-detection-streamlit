import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="centered"
)

st.markdown("## 🚦 Vehicle Detection App")
st.markdown("### 🎯 Upload a vehicle video and download the processed result")
st.markdown("**👨‍💻 Created by Gurudeep Soni**")
st.markdown("---")

video = st.file_uploader("📤 Upload vehicle video", type=["mp4", "avi"])

if video:
    # Save uploaded video
    input_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_video.write(video.read())
    input_video.close()

    st.info("⏳ Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_video.name)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video.name, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status = st.empty()

    frame_count = 0
    st.success("🚀 Processing started...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        out.write(results[0].plot())

        frame_count += 1
        if total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            status.text(f"🔄 Processing: {progress}%")

    cap.release()
    out.release()

    progress_bar.progress(100)
    status.text("✅ Processing complete!")

    st.markdown("### 📥 Download Result")

    with open(output_video.name, "rb") as f:
        st.download_button(
            label="⬇️ Download processed video",
            data=f,
            file_name="vehicle_detection_result.mp4",
            mime="video/mp4"
        )

    st.success("🎉 Done! Video is ready to download.")
