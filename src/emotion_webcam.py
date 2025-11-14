# emotion_webcam.py
from deepface import DeepFace
import cv2
from utils import draw_emotion

cap = cv2.VideoCapture(0)
print("Starting advanced webcam emotion detector... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
        frame = draw_emotion(frame, emotion, confidence)
    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
