#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
import json
import os
from queue import Queue, Empty
from rknnlite.api import RKNNLite

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RKNN_MODEL_PATH = "yolov8s.rknn"
RTSP_URL = "rtsp://admin:123456Ai@192.168.1.16:554/snl/live/1/1"

CONFIG_PATH = "config.json"

INPUT_SIZE = (640, 640)
CORE_MASK = RKNNLite.NPU_CORE_AUTO

# YOLO CLASSES
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
# CONFIG LOADER (HOT RELOAD)
# ──────────────────────────────────────────────
last_mtime = 0
config = {}

def get_config():
    global last_mtime, config
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
        if mtime != last_mtime:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            last_mtime = mtime
            print("[INFO] Config reloaded")
    except:
        pass
    return config

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
# POSTPROCESS
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

def postprocess(outputs, shape, scale, pad):
    cfg = get_config()

    CONF = cfg.get("confidence_threshold", 0.45)
    IOU  = cfg.get("iou_threshold", 0.45)
    enabled = set(cfg.get("enabled_classes", COCO_CLASSES))

    pred = outputs[0][0].T
    boxes = pred[:, :4]
    scores = pred[:, 4:]

    conf = scores.max(axis=1)
    cls  = scores.argmax(axis=1)

    mask = conf >= CONF
    boxes, conf, cls = boxes[mask], conf[mask], cls[mask]

    if len(conf) == 0:
        return [], [], []

    # scale back
    boxes[:, 0] = (boxes[:, 0] - pad[0]) / scale
    boxes[:, 1] = (boxes[:, 1] - pad[1]) / scale
    boxes[:, 2:] /= scale

    # xywh → xyxy
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    # CLASS FILTER
    filtered = [
        (b, s, c)
        for b, s, c in zip(boxes, conf, cls)
        if COCO_CLASSES[int(c)] in enabled
    ]

    if not filtered:
        return [], [], []

    boxes, conf, cls = map(np.array, zip(*filtered))

    # 🔥 NMS FIX
    keep = nms(boxes, conf, IOU)

    return boxes[keep], conf[keep], cls[keep]

# ──────────────────────────────────────────────
# CAPTURE THREAD (REAL-TIME FIX)
# ──────────────────────────────────────────────
def capture_thread(rtsp_url, raw_q, stop_evt):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        # 🔥 CRITICAL: DROP OLD FRAMES
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
    rknn = RKNNLite()
    rknn.load_rknn(RKNN_MODEL_PATH)
    rknn.init_runtime(core_mask=CORE_MASK)

    raw_q = Queue(maxsize=1)   # 🔥 important
    res_q = Queue(maxsize=1)

    stop = threading.Event()

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

        cfg = get_config()

        for b, s, c in zip(boxes, scores, cls):
            x1, y1, x2, y2 = map(int, b)
            label = ""

            if cfg.get("show_labels", True):
                label += COCO_CLASSES[int(c)]

            if cfg.get("show_confidence", True):
                label += f" {s:.2f}"

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

            if label:
                cv2.putText(frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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