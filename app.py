import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.title("Vehicle Detection App")

video = st.file_uploader("Upload vehicle video", type=["mp4", "avi"])

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(tfile.name)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    st.text("Processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        out.write(results[0].plot())

    cap.release()
    out.release()

    st.video("result.mp4")
