from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot access webcam.")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error:", e)

    cv2.imshow("Real-Time Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
