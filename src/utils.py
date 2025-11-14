# utils.py
import cv2

# Color mapping for different emotions
EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 255, 255),
    "fear": (128, 0, 128),
    "disgust": (0, 128, 0),
    "neutral": (200, 200, 200),
}

def draw_emotion(frame, emotion, confidence):
    """Draws the emotion text and confidence on the frame."""
    color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
    text = f"{emotion.capitalize()} ({confidence:.1f}%)"
    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return frame
