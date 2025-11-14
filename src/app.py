# app.py
import streamlit as st
import cv2
from deepface import DeepFace
import tempfile
import os
from utils import draw_emotion

st.set_page_config(page_title="AI Emotion Detector", layout="centered")

st.title("🤖  Jarvis Real-Time Emotion Detection")
st.markdown("Detects your emotion live using your webcam.")

# Webcam capture section
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
st.markdown("<hr>", unsafe_allow_html=True)

# Function for emotion detection
def analyze_emotion(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    confidence = result[0]['emotion'][emotion]
    return emotion, confidence

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to access camera.")
            break
        try:
            emotion, confidence = analyze_emotion(frame)
            frame = draw_emotion(frame, emotion, confidence)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        except Exception as e:
            st.write("Error:", e)
        if not run:
            break
    cap.release()
else:
    st.info("✅ Click the checkbox above to start the webcam.")
