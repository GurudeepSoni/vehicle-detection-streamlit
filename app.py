import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚦 Vehicle Detection App")
st.write("Upload a video and download processed output")

video = st.file_uploader("Upload video", type=["mp4", "avi"])

if video:
    # Save input
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_file.write(video.read())
    input_file.close()

    st.info("Loading model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_file.name)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_file.close()

    out = cv2.VideoWriter(
        output_file.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    progress = st.progress(0)
    frame_count = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        out.write(results[0].plot())

        frame_count += 1
        if total:
            progress.progress(int(frame_count / total * 100))

    cap.release()
    out.release()

    st.success("Done!")

    with open(output_file.name, "rb") as f:
        st.download_button(
            "Download Result",
            f,
            file_name="output.mp4"
        )
