#!/usr/bin/env python3
"""
YOLOv8 RKNN Inference on Video — Optimised Edition
- Multi-threaded capture + inference pipeline (doubles effective FPS)
- Display window capped to screen / custom size
- Frame skipping when inference falls behind
Requirements: rknn-toolkit-lite2, opencv-python, numpy
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
RKNN_MODEL_PATH  = "yolov8s.rknn"
VIDEO_PATH       = "test_video.mp4"
OUTPUT_PATH      = None                   # e.g. "output.mp4" or None to skip saving

INPUT_SIZE       = (640, 640)
CONF_THRESHOLD   = 0.45
IOU_THRESHOLD    = 0.45
NUM_CLASSES      = 80
CORE_MASK        = RKNNLite.NPU_CORE_AUTO

# ── Display window constraints ────────────────
MAX_DISPLAY_W    = 1280   # cap window width  (pixels)
MAX_DISPLAY_H    = 720    # cap window height (pixels)

# ── Pipeline tuning ───────────────────────────
CAPTURE_QUEUE_SIZE  = 4   # raw frames buffered ahead of inference
RESULT_QUEUE_SIZE   = 2   # annotated frames buffered before display
SKIP_FRAMES_THRESH  = 3   # drop oldest raw frame when queue exceeds this

# COCO class names
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

np.random.seed(42)
CLASS_COLORS = {i: tuple(int(c) for c in np.random.randint(50, 230, 3))
                for i in range(NUM_CLASSES)}

# ──────────────────────────────────────────────
# PRE / POST PROCESSING
# ──────────────────────────────────────────────

def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    nw_u, nh_u = int(w * scale), int(h * scale)
    img_r = cv2.resize(img, (nw_u, nh_u), interpolation=cv2.INTER_LINEAR)
    dw = (nw - nw_u) / 2
    dh = (nh - nh_u) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_p = cv2.copyMakeBorder(img_r, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return img_p, scale, (dw, dh)


def xywh2xyxy(b):
    o = np.copy(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou  = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def postprocess(outputs, orig_shape, scale, pad):
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T                          # (8400, 84)

    boxes_raw    = pred[:, :4]
    class_scores = pred[:, 4:]
    conf         = class_scores.max(axis=1)
    class_ids    = class_scores.argmax(axis=1)

    mask      = conf >= CONF_THRESHOLD
    boxes_raw = boxes_raw[mask]
    conf      = conf[mask]
    class_ids = class_ids[mask]

    if len(conf) == 0:
        return [], [], []

    dw, dh = pad
    boxes_raw[:, 0] = (boxes_raw[:, 0] - dw) / scale
    boxes_raw[:, 1] = (boxes_raw[:, 1] - dh) / scale
    boxes_raw[:, 2] =  boxes_raw[:, 2]        / scale
    boxes_raw[:, 3] =  boxes_raw[:, 3]        / scale

    boxes_xyxy = xywh2xyxy(boxes_raw)
    oh, ow = orig_shape[:2]
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, ow)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, oh)

    keep = nms(boxes_xyxy, conf, IOU_THRESHOLD)
    return boxes_xyxy[keep], conf[keep], class_ids[keep]

# ──────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────

def draw_detections(frame, boxes, scores, class_ids, disp_scale):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = (int(v * disp_scale) for v in box)
        color = CLASS_COLORS.get(int(cls_id), (0, 255, 0))
        name  = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else str(cls_id)
        label = f"{name} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 2, ly), color, -1)
        cv2.putText(frame, label, (x1 + 1, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps, n_det):
    cv2.putText(frame, f"FPS: {fps:5.1f}",  (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Dets: {n_det}",    (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

# ──────────────────────────────────────────────
# COMPUTE DISPLAY SCALE
# ──────────────────────────────────────────────

def compute_display_scale(src_w, src_h):
    scale  = min(MAX_DISPLAY_W / src_w, MAX_DISPLAY_H / src_h, 1.0)
    disp_w = int(src_w * scale)
    disp_h = int(src_h * scale)
    return disp_w, disp_h, scale

# ──────────────────────────────────────────────
# PIPELINE THREADS
# ──────────────────────────────────────────────

def capture_thread(cap, raw_q, stop_evt):
    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            raw_q.put(None)
            break
        # drop stale frames to stay live
        while raw_q.qsize() >= SKIP_FRAMES_THRESH:
            try: raw_q.get_nowait()
            except Empty: break
        raw_q.put(frame)


def inference_thread(rknn, raw_q, res_q, stop_evt):
    while not stop_evt.is_set():
        try:
            frame = raw_q.get(timeout=1.0)
        except Empty:
            continue
        if frame is None:
            res_q.put(None)
            break
        img, scale, pad = letterbox(frame)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = rknn.inference(inputs=[np.expand_dims(img_rgb, 0)])
        boxes, scores, class_ids = postprocess(outputs, frame.shape, scale, pad)
        res_q.put((frame, boxes, scores, class_ids))

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("[INFO] Loading RKNN model …")
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(RKNN_MODEL_PATH) != 0:
        raise RuntimeError(f"Cannot load {RKNN_MODEL_PATH}")
    if rknn.init_runtime(core_mask=CORE_MASK) != 0:
        raise RuntimeError("Cannot init RKNN runtime")
    print("[INFO] Model ready")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    disp_w, disp_h, disp_scale = compute_display_scale(src_w, src_h)

    print(f"[INFO] Source  : {src_w}×{src_h} @ {src_fps:.1f} fps  ({total} frames)")
    print(f"[INFO] Display : {disp_w}×{disp_h}  (scale {disp_scale:.3f})")

    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, src_fps, (src_w, src_h))

    raw_q = Queue(maxsize=CAPTURE_QUEUE_SIZE)
    res_q = Queue(maxsize=RESULT_QUEUE_SIZE)
    stop  = threading.Event()

    t_cap = threading.Thread(target=capture_thread,
                             args=(cap, raw_q, stop), daemon=True)
    t_inf = threading.Thread(target=inference_thread,
                             args=(rknn, raw_q, res_q, stop), daemon=True)
    t_cap.start()
    t_inf.start()

    # Fixed-size window — will NOT resize beyond MAX bounds
    cv2.namedWindow("YOLOv8 RKNN", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 RKNN", disp_w, disp_h)

    fps_smooth = src_fps
    alpha      = 0.1
    delay_ms   = max(1, int(1000 / src_fps))
    frame_idx  = 0
    t_last     = time.perf_counter()

    while True:
        try:
            item = res_q.get(timeout=2.0)
        except Empty:
            break
        if item is None:
            break

        frame, boxes, scores, class_ids = item
        frame_idx += 1

        if writer:
            ann = draw_detections(frame.copy(), boxes, scores, class_ids, 1.0)
            writer.write(ann)

        # resize to display resolution FIRST, then draw (fast path)
        disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        draw_detections(disp, boxes, scores, class_ids, disp_scale)

        now = time.perf_counter()
        fps_smooth = alpha * (1.0 / max(now - t_last, 1e-6)) + (1 - alpha) * fps_smooth
        t_last = now
        draw_hud(disp, fps_smooth, len(boxes))

        cv2.imshow("YOLOv8 RKNN", disp)

        spent = int((time.perf_counter() - t_last) * 1000)
        wait  = max(1, delay_ms - spent)
        key   = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("[INFO] Quit")
            stop.set()
            break

        if frame_idx % 60 == 0:
            print(f"  frame {frame_idx}/{total}  FPS {fps_smooth:.1f}  dets {len(boxes)}")

    stop.set()
    t_cap.join(timeout=2)
    t_inf.join(timeout=2)
    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Saved: {OUTPUT_PATH}")
    cv2.destroyAllWindows()
    rknn.release()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()