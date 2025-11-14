# detect_image.py
try:
	from deepface import DeepFace
except Exception:
	print("Missing dependency 'deepface'. Install it with:\n    python -m pip install deepface")
	raise SystemExit(1)

import cv2
try:
	import matplotlib.pyplot as plt
	_HAS_MATPLOTLIB = True
except Exception:
	_HAS_MATPLOTLIB = False

# Step 1: Load image
img_path = "images/face.jpg"  # Path to your image
print(f"Analyzing image: {img_path}")

# Step 2: Analyze emotions using DeepFace
result = DeepFace.analyze(img_path=img_path, actions=['emotion'])

# Extract dominant emotion from the result (handle dict or list return types)
if isinstance(result, list) and len(result) > 0:
	first = result[0]
	dominant_emotion = first.get('dominant_emotion') or (first.get('emotion') and first['emotion'].get('dominant_emotion')) or ""
else:
	dominant_emotion = result.get('dominant_emotion') if isinstance(result, dict) else ""
if not dominant_emotion:
	# As a fallback, try to inspect nested 'emotion' dict
	if isinstance(result, dict) and result.get('emotion'):
		dominant_emotion = result['emotion'].get('dominant_emotion', '')
print(f"Detected dominant emotion: {dominant_emotion}")

# Step 4: Display image with emotion label
img = cv2.imread(img_path)
if img is None:
	print("Failed to load image:", img_path)
else:
	if _HAS_MATPLOTLIB:
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		plt.title(f"Emotion: {dominant_emotion}")
		plt.axis("off")
		plt.show()
	else:
		# Fallback to OpenCV window (works if a GUI is available)
		cv2.putText(img, f"Emotion: {dominant_emotion}", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.imshow("Emotion", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
