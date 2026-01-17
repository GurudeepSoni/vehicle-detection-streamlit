import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="centered"
)

st.markdown("## 🚦 Vehicle Detection App")
st.markdown("### 🎯 Upload a vehicle video and get detection result")
st.markdown("**👨‍💻 Created by Gurudeep Soni**")
st.markdown("---")

# ---------------- Upload ----------------
video = st.file_uploader("📤 Upload vehicle video", type=["mp4", "avi"])

if video:
    # Save uploaded video to temp file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video.read())
    temp_input.close()

    st.info("⏳ Loading model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(temp_input.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # fallback for Streamlit Cloud

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video as temp file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status = st.empty()

    frame_count = 0
    st.success("🚀 Processing started...")

    # ---------------- Processing ----------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1
        if total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            status.text(f"🔄 Processing: {progress}%")

    cap.release()
    out.release()

    progress_bar.progress(100)
    status.text("✅ Processing complete!")

    st.markdown("### 🎥 Output Video")

    # ---------------- Display video ----------------
    if os.path.exists(temp_output.name) and os.path.getsize(temp_output.name) > 0:
        with open(temp_output.name, "rb") as f:
            st.video(f.read())
    else:
        st.error("❌ Output video could not be generated.")

    st.success("🎉 Done! Thanks for using the app.")
