import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# ---------- UI ----------
st.set_page_config(page_title="Vehicle Detection", page_icon="🚗", layout="centered")

st.markdown("## 🚦 Vehicle Detection App")
st.markdown("### 🎯 Upload a vehicle video and get detection result")
st.markdown("**👨‍💻 Created by Gurudeep Soni**")
st.markdown("---")

video = st.file_uploader("📤 Upload vehicle video", type=["mp4", "avi"])

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    st.info("⏳ Loading model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    current = 0
    st.success("🚀 Processing started...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        out.write(results[0].plot())

        current += 1
        progress = int((current / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"🔄 Processing: {progress}%")

    cap.release()
    out.release()

    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")

    st.markdown("### 🎥 Output Video")
    st.video("result.mp4")

    st.success("🎉 Done! Thanks for using the app.")
