#!/usr/bin/env python3
"""
YOLOv8 RKNN Inference on RTSP Stream (Low Latency, Stable)
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from rknnlite.api import RKNNLite

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RKNN_MODEL_PATH = "yolov8s.rknn"

# 🔴 Replace this with your RTSP stream
RTSP_URL = "rtsp://admin:123456Ai@192.168.1.16:554/snl/live/1/1"

INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
CORE_MASK = RKNNLite.NPU_CORE_AUTO

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

# ──────────────────────────────────────────────
# PREPROCESS
# ──────────────────────────────────────────────
def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    nw_u, nh_u = int(w * scale), int(h * scale)
    img_r = cv2.resize(img, (nw_u, nh_u))
    dw = (nw - nw_u) / 2
    dh = (nh - nh_u) / 2
    top, bottom = int(dh), int(dh)
    left, right = int(dw), int(dw)
    return cv2.copyMakeBorder(img_r, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color), scale, (dw, dh)

def xywh2xyxy(b):
    o = np.copy(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o

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

def postprocess(outputs, orig_shape, scale, pad):
    pred = outputs[0][0].T
    boxes = pred[:, :4]
    scores = pred[:, 4:]

    conf = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = conf >= CONF_THRESHOLD
    boxes, conf, class_ids = boxes[mask], conf[mask], class_ids[mask]

    if len(conf) == 0:
        return [], [], []

    dw, dh = pad
    boxes[:, 0] = (boxes[:, 0] - dw) / scale
    boxes[:, 1] = (boxes[:, 1] - dh) / scale
    boxes[:, 2:] /= scale

    boxes = xywh2xyxy(boxes)

    keep = nms(boxes, conf, IOU_THRESHOLD)
    return boxes[keep], conf[keep], class_ids[keep]

# ──────────────────────────────────────────────
# CAPTURE THREAD (RTSP optimized)
# ──────────────────────────────────────────────
def capture_thread(rtsp_url, raw_q, stop_evt):
    while not stop_evt.is_set():
        print("[INFO] Connecting to RTSP...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        # Reduce buffering → lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cap.isOpened():
            print("[WARN] RTSP connection failed, retrying...")
            time.sleep(2)
            continue

        print("[INFO] RTSP connected")

        while not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Stream lost, reconnecting...")
                break

            if not raw_q.full():
                raw_q.put(frame)

        cap.release()

# ──────────────────────────────────────────────
# INFERENCE THREAD
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
        boxes, scores, class_ids = postprocess(outputs, frame.shape, scale, pad)

        res_q.put((frame, boxes, scores, class_ids))

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    rknn = RKNNLite()
    rknn.load_rknn(RKNN_MODEL_PATH)
    rknn.init_runtime(core_mask=CORE_MASK)

    raw_q = Queue(maxsize=4)
    res_q = Queue(maxsize=2)
    stop = threading.Event()

    threading.Thread(target=capture_thread,
                     args=(RTSP_URL, raw_q, stop),
                     daemon=True).start()

    threading.Thread(target=inference_thread,
                     args=(rknn, raw_q, res_q, stop),
                     daemon=True).start()

    cv2.namedWindow("RTSP RKNN", cv2.WINDOW_NORMAL)

    t_last = time.time()

    while True:
        try:
            frame, boxes, scores, class_ids = res_q.get(timeout=5)
        except Empty:
            print("[WARN] No frames received")
            break

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

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