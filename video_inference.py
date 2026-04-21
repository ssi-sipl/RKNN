import cv2
import numpy as np
import requests
from rknnlite.api import RKNNLite

MODEL_PATH = "yolov8s.rknn"
VIDEO_PATH = "test_video.mp4"

# -------------------------
# Download test video
# -------------------------
print("Downloading test video...")
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.mp4"

try:
    r = requests.get(url)
    open(VIDEO_PATH, "wb").write(r.content)
    print("Video downloaded")
except:
    print("Video already exists or download failed")

# -------------------------
# Load RKNN model
# -------------------------
print("Loading RKNN model...")

rknn = RKNNLite()

ret = rknn.load_rknn(MODEL_PATH)
if ret != 0:
    print("Failed to load model")
    exit(ret)

ret = rknn.init_runtime()
if ret != 0:
    print("Failed to init runtime")
    exit(ret)

print("Model ready")

# -------------------------
# YOLO helper
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------
# Open video
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Failed to open video")
    exit()

print("Starting inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640,640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_input = np.expand_dims(img_rgb, 0)

    outputs = rknn.inference(inputs=[img_input])

    # simple visualization placeholder
    cv2.putText(frame,
                "RKNN inference running",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("RKNN YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

rknn.release()
