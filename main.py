#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
import argparse
from queue import Queue, Empty
from rknnlite.api import RKNNLite
import os

# ──────────────────────────────────────────────
# CLI ARGUMENTS (CONTROLLED FROM NODE)
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()

parser.add_argument("--rtsp", required=True)
parser.add_argument("--conf", type=float, default=0.45)
parser.add_argument("--iou", type=float, default=0.45)
parser.add_argument("--show_labels", type=int, default=1)
parser.add_argument("--show_conf", type=int, default=1)

args = parser.parse_args()

RTSP_URL = args.rtsp
CONF_THRESHOLD = args.conf
IOU_THRESHOLD = args.iou
SHOW_LABELS = bool(args.show_labels)
SHOW_CONF = bool(args.show_conf)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RKNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s.rknn")
INPUT_SIZE = (640, 640)
CORE_MASK = RKNNLite.NPU_CORE_AUTO

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ──────────────────────────────────────────────
# PREPROCESS
# ──────────────────────────────────────────────
def letterbox(img):
    h, w = img.shape[:2]
    scale = min(640 / w, 640 / h)
    nh, nw = int(h * scale), int(w * scale)
    img_r = cv2.resize(img, (nw, nh))
    top = (640 - nh) // 2
    left = (640 - nw) // 2
    img_p = cv2.copyMakeBorder(img_r, top, 640-nh-top, left, 640-nw-left,
                               cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_p, scale, (left, top)

# ──────────────────────────────────────────────
# NMS
# ──────────────────────────────────────────────
def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return keep

# ──────────────────────────────────────────────
# POSTPROCESS
# ──────────────────────────────────────────────
def postprocess(outputs, shape, scale, pad):
    pred = outputs[0][0].T
    boxes = pred[:, :4]
    scores = pred[:, 4:]

    conf = scores.max(axis=1)
    cls  = scores.argmax(axis=1)

    mask = conf >= CONF_THRESHOLD
    boxes, conf, cls = boxes[mask], conf[mask], cls[mask]

    if len(conf) == 0:
        return [], [], []

    boxes[:, 0] = (boxes[:, 0] - pad[0]) / scale
    boxes[:, 1] = (boxes[:, 1] - pad[1]) / scale
    boxes[:, 2:] /= scale

    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    keep = nms(boxes, conf, IOU_THRESHOLD)

    return boxes[keep], conf[keep], cls[keep]

# ──────────────────────────────────────────────
# SIMPLE TRACKER
# ──────────────────────────────────────────────
class SimpleTracker:
    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}
        self.lost = {}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, detections):
        updated = {}

        for tid, tbox in self.tracks.items():
            best_iou = 0
            best = None
            for d in detections:
                iou_val = self.iou(tbox, d)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best = d

            if best_iou > self.iou_threshold:
                updated[tid] = best
                self.lost[tid] = 0
            else:
                self.lost[tid] += 1

        self.tracks = {
            tid: box for tid, box in updated.items()
            if self.lost.get(tid, 0) <= self.max_lost
        }

        for d in detections:
            matched = any(self.iou(d, t) > self.iou_threshold for t in self.tracks.values())
            if not matched:
                self.tracks[self.next_id] = d
                self.lost[self.next_id] = 0
                self.next_id += 1

        return self.tracks

# ──────────────────────────────────────────────
# CAPTURE
# ──────────────────────────────────────────────
def capture_thread(rtsp_url, raw_q, stop_evt):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        while not raw_q.empty():
            try:
                raw_q.get_nowait()
            except:
                break

        raw_q.put(frame)

# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def inference_thread(rknn, raw_q, res_q, stop_evt):
    while not stop_evt.is_set():
        try:
            frame = raw_q.get(timeout=1)
        except Empty:
            continue

        img, scale, pad = letterbox(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = rknn.inference(inputs=[np.expand_dims(img, 0)])
        boxes, scores, cls = postprocess(outputs, frame.shape, scale, pad)

        res_q.put((frame, boxes, scores, cls))

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"[INFO] Starting inference on: {RTSP_URL}")

    rknn = RKNNLite()
    rknn.load_rknn(RKNN_MODEL_PATH)
    rknn.init_runtime(core_mask=CORE_MASK)

    raw_q = Queue(maxsize=1)
    res_q = Queue(maxsize=1)

    stop = threading.Event()
    tracker = SimpleTracker()

    threading.Thread(target=capture_thread,
                     args=(RTSP_URL, raw_q, stop),
                     daemon=True).start()

    threading.Thread(target=inference_thread,
                     args=(rknn, raw_q, res_q, stop),
                     daemon=True).start()

    t_last = time.time()

    while True:
        try:
            frame, boxes, scores, cls = res_q.get(timeout=5)
        except Empty:
            break

        tracks = tracker.update(boxes)

        for tid, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)

            label = f"ID {tid}"

            for b, s, c in zip(boxes, scores, cls):
                if tracker.iou(box, b) > 0.5:
                    if SHOW_LABELS:
                        label += f" {COCO_CLASSES[int(c)]}"
                    if SHOW_CONF:
                        label += f" {s:.2f}"
                    break

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

        fps = 1 / (time.time() - t_last)
        t_last = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("RTSP RKNN", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop.set()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == "__main__":
    main()