#!/usr/bin/env python3
"""
YOLOv8 RKNN Inference on Video
Requirements: rknn-toolkit-lite2, opencv-python, numpy
"""

import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

# ──────────────────────────────────────────────
# CONFIG — edit these as needed
# ──────────────────────────────────────────────
RKNN_MODEL_PATH = "yolov8s.rknn"
VIDEO_PATH      = "test_video.mp4"
OUTPUT_PATH     = "output_detected.mp4"   # set to None to skip saving
INPUT_SIZE      = (640, 640)              # model input resolution
CONF_THRESHOLD  = 0.45                   # confidence threshold
IOU_THRESHOLD   = 0.45                   # NMS IoU threshold
NUM_CLASSES     = 80                     # COCO default; change if custom model
CORE_MASK       = RKNNLite.NPU_CORE_AUTO # NPU core selection

# COCO class names (80 classes)
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

# Generate a fixed colour per class
np.random.seed(42)
CLASS_COLORS = {i: tuple(int(c) for c in np.random.randint(50, 255, 3)) for i in range(NUM_CLASSES)}

# ──────────────────────────────────────────────
# PRE/POST PROCESSING HELPERS
# ──────────────────────────────────────────────

def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    """Resize with padding to maintain aspect ratio."""
    h, w = img.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    nw_unpad, nh_unpad = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw_unpad, nh_unpad), interpolation=cv2.INTER_LINEAR)

    dw = (nw - nw_unpad) / 2
    dh = (nh - nh_unpad) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, (dw, dh)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(boxes):
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes, scores, iou_thresh):
    """Simple NMS returning kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou  = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def postprocess(outputs, orig_shape, scale, pad):
    """
    Parse raw RKNN output for YOLOv8 (single-output head).
    Expected output shape: (1, 84, 8400)  — adjust if your export differs.
    """
    pred = outputs[0]  # shape: (1, 84, 8400) or (84, 8400)
    if pred.ndim == 3:
        pred = pred[0]                   # → (84, 8400)
    pred = pred.T                        # → (8400, 84)

    boxes_raw = pred[:, :4]
    class_scores = pred[:, 4:]          # (8400, 80)

    conf = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = conf >= CONF_THRESHOLD
    boxes_raw = boxes_raw[mask]
    conf      = conf[mask]
    class_ids = class_ids[mask]

    if len(conf) == 0:
        return [], [], []

    # Scale boxes from model space → original image space
    dw, dh = pad
    boxes_raw[:, 0] = (boxes_raw[:, 0] - dw) / scale
    boxes_raw[:, 1] = (boxes_raw[:, 1] - dh) / scale
    boxes_raw[:, 2] = boxes_raw[:, 2] / scale
    boxes_raw[:, 3] = boxes_raw[:, 3] / scale

    boxes_xyxy = xywh2xyxy(boxes_raw)

    # Clip to image bounds
    oh, ow = orig_shape[:2]
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, ow)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, oh)

    keep = nms(boxes_xyxy, conf, IOU_THRESHOLD)
    return boxes_xyxy[keep], conf[keep], class_ids[keep]


# ──────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────

def draw_detections(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(int(cls_id), (0, 255, 0))
        label = f"{COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else cls_id} {score:.2f}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1, th + 6)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 2, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 1, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps, n_det):
    h = frame.shape[0]
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {n_det}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    # ── Load RKNN model ──────────────────────
    print("[INFO] Loading RKNN model …")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {RKNN_MODEL_PATH}")

    ret = rknn.init_runtime(core_mask=CORE_MASK)
    if ret != 0:
        raise RuntimeError("Failed to init RKNN runtime")
    print("[INFO] RKNN model loaded OK")

    # ── Open video ───────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {src_w}×{src_h} @ {src_fps:.1f} fps  ({total} frames)")

    # ── Optional video writer ─────────────────
    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, src_fps, (src_w, src_h))

    # ── Frame-timing for smooth display ──────
    delay_ms        = max(1, int(1000 / src_fps))
    fps_smooth      = src_fps
    alpha           = 0.1                # EMA smoothing factor
    frame_idx       = 0

    print("[INFO] Starting inference — press  Q  to quit")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Pre-process ───────────────────────
        img, scale, pad = letterbox(frame)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img_rgb, 0)   # (1, 640, 640, 3)

        # ── Inference ────────────────────────
        outputs = rknn.inference(inputs=[inp])

        # ── Post-process ──────────────────────
        boxes, scores, class_ids = postprocess(outputs, frame.shape, scale, pad)

        # ── Draw ─────────────────────────────
        if len(boxes):
            frame = draw_detections(frame, boxes, scores, class_ids)

        # ── FPS (EMA) ─────────────────────────
        elapsed  = time.perf_counter() - t0
        inst_fps = 1.0 / max(elapsed, 1e-6)
        fps_smooth = alpha * inst_fps + (1 - alpha) * fps_smooth

        frame = draw_hud(frame, fps_smooth, len(boxes))

        # ── Save frame ───────────────────────
        if writer:
            writer.write(frame)

        # ── Display (smooth timing) ───────────
        cv2.imshow("YOLOv8 RKNN Inference", frame)

        # Spend leftover time waiting so playback matches source FPS
        spent_ms = int((time.perf_counter() - t0) * 1000)
        wait     = max(1, delay_ms - spent_ms)
        if cv2.waitKey(wait) & 0xFF in (ord("q"), ord("Q"), 27):
            print("[INFO] Quit by user")
            break

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{total}  |  FPS {fps_smooth:.1f}  |  dets {len(boxes)}")

    # ── Cleanup ───────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Saved output to: {OUTPUT_PATH}")
    cv2.destroyAllWindows()
    rknn.release()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()