import streamlit as st
import cv2
from fer.fer import FER
import numpy as np

st.set_page_config(page_title="AI Emotion Detector", layout="centered")

st.title("😊 Real-Time Emotion Detector (Streamlit Compatible)")
st.markdown("Works on Streamlit Cloud • Browser Webcam Support • No DeepFace/TensorFlow")

detector = FER()

st.markdown("### 📸 Capture your face")
img = st.camera_input("Turn on camera and take a picture")

if img:
    # Convert image to array
    img_bytes = img.getvalue()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert BGR → RGB for FER
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotion
    emotions = detector.detect_emotions(rgb)

    if emotions:
        box = emotions[0]["box"]
        emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])

        x, y, w, h = box

        # Draw on image
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb, f"{emotion} ({score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        st.image(rgb, caption=f"Detected emotion: {emotion}")

    else:
        st.warning("No face detected. Please try again.")
