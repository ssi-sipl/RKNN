import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time

MODEL_PATH = "yolov8s.rknn"
VIDEO_PATH = "test_video.mp4"

INPUT_SIZE = 640
CONF_THRES = 0.4
IOU_THRES = 0.45

CLASSES = [
'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
'remote','keyboard','cell phone','microwave','oven','toaster','sink',
'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# RKNN INIT

rknn = RKNNLite()
print("Loading model...")
rknn.load_rknn(MODEL_PATH)
rknn.init_runtime()
print("Model ready")

# NMS

def nms(boxes, scores, iou_threshold):
indices = cv2.dnn.NMSBoxes(
boxes,
scores,
CONF_THRES,
iou_threshold
)
return indices

# Decode YOLOv8 output

def decode(outputs):

```
boxes = []
scores = []
class_ids = []

pred = outputs[0]

for det in pred[0]:

    score = np.max(det[4:])
    class_id = np.argmax(det[4:])

    if score > CONF_THRES:

        x, y, w, h = det[0:4]

        x1 = int(x - w/2)
        y1 = int(y - h/2)

        boxes.append([x1, y1, int(w), int(h)])
        scores.append(float(score))
        class_ids.append(class_id)

idx = nms(boxes, scores, IOU_THRES)

final = []

if len(idx) > 0:
    for i in idx.flatten():
        final.append((boxes[i], scores[i], class_ids[i]))

return final
```

cap = cv2.VideoCapture(VIDEO_PATH)

fps_time = time.time()

while True:

```
ret, frame = cap.read()
if not ret:
    break

img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.expand_dims(img, 0)

outputs = rknn.inference(inputs=[img])

detections = decode(outputs)

for box, score, class_id in detections:

    x, y, w, h = box

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    label = f"{CLASSES[class_id]} {score:.2f}"

    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2)

fps = 1/(time.time()-fps_time)
fps_time = time.time()

cv2.putText(frame, f"FPS {fps:.2f}", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow("RKNN YOLOv8", frame)

if cv2.waitKey(1) == 27:
    break
```

cap.release()
cv2.destroyAllWindows()
rknn.release()
