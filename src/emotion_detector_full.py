# emotion_detector_full.py
from deepface import DeepFace
import cv2
from utils import draw_emotion
import csv
from datetime import datetime

LOG_FILE = "../logs/emotion_log.csv"

# Create log file header if not exists
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Emotion", "Confidence"])

cap = cv2.VideoCapture(0)
print("Full Emotion Detector Running... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
        frame = draw_emotion(frame, emotion, confidence)

        # Log detected emotion
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, confidence])

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Detector - Full Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
