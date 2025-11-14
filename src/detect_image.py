# detect_image.py

# -----------------------------
# Imports
# -----------------------------
try:
    from deepface import DeepFace
except Exception:
    print("Missing dependency 'deepface'. Install it using:\n\n    pip install deepface\n")
    raise SystemExit(1)

import cv2

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


# -----------------------------
# Step 1: Load Image
# -----------------------------
img_path = "images/face.jpg"   # Change this to your image
print(f"\nAnalyzing image: {img_path}")

# -----------------------------
# Step 2: Analyze Emotions
# -----------------------------
try:
    result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
except Exception as e:
    print("DeepFace failed to analyze the image.")
    print("Error:", e)
    raise SystemExit(1)


# -----------------------------
# Step 3: Extract Dominant Emotion
# -----------------------------
dominant_emotion = ""

# DeepFace sometimes returns a list, sometimes a dict → handle both
if isinstance(result, list) and len(result) > 0:
    first = result[0]
    dominant_emotion = first.get("dominant_emotion") or (
        first.get("emotion") and first["emotion"].get("dominant_emotion")
    )
elif isinstance(result, dict):
    dominant_emotion = result.get("dominant_emotion") or (
        result.get("emotion") and result["emotion"].get("dominant_emotion")
    )

if not dominant_emotion:
    dominant_emotion = "Unknown"

print(f"Detected dominant emotion: {dominant_emotion}")

# -----------------------------
# Step 4: Display Image with Label
# -----------------------------
img = cv2.imread(img_path)

if img is None:
    print("\n❌ ERROR: Could not load image:", img_path)
else:
    if _HAS_MATPLOTLIB:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Emotion: {dominant_emotion}")
        plt.axis("off")
        plt.show()
    else:
        # OpenCV fallback (desktop only)
        cv2.putText(img, f"Emotion: {dominant_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
