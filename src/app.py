import streamlit as st
import cv2
from fer.fer import FER   # ✅ FIXED IMPORT
import numpy as np

st.set_page_config(page_title="AI Emotion Detector", layout="centered")

st.title("😊 Real-Time Emotion Detection (Streamlit Compatible)")
st.markdown("No sound • No TensorFlow • No DeepFace • Works on Streamlit Cloud")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

detector = FER()

camera = None

if run:
    camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera not accessible")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    emotions = detector.detect_emotions(rgb)

    if emotions:
        box = emotions[0]["box"]
        emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])

        x, y, w, h = box
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(rgb, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    FRAME_WINDOW.image(rgb)

if camera:
    camera.release()
