#!/usr/bin/env python3

import os
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;1024000"
)

import cv2
import time
import threading
import argparse
import numpy as np
from rknnlite.api import RKNNLite

# --------------------------------
# CLI
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--rtsp", required=True)
parser.add_argument("--conf", type=float, default=0.45)
parser.add_argument("--iou", type=float, default=0.45)

args = parser.parse_args()

RTSP = args.rtsp
CONF = args.conf
IOU = args.iou

# --------------------------------
# CONFIG
# --------------------------------

MODEL = "yolov8s.rknn"
INPUT_SIZE = 640
CORE = RKNNLite.NPU_CORE_0

running = True

latest_frame = None
latest_result = None

frame_lock = threading.Lock()
result_lock = threading.Lock()

COCO_CLASSES = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
"umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
"bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
"couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
"remote","keyboard","cell phone","microwave","oven","toaster","sink",
"refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
"toothbrush"
]

# --------------------------------
# LETTERBOX
# --------------------------------

def letterbox(img):

    h,w = img.shape[:2]

    scale=min(
        INPUT_SIZE/w,
        INPUT_SIZE/h
    )

    nw=int(w*scale)
    nh=int(h*scale)

    resized=cv2.resize(
        img,
        (nw,nh)
    )

    canvas=np.full(
        (INPUT_SIZE,INPUT_SIZE,3),
        114,
        dtype=np.uint8
    )

    left=(INPUT_SIZE-nw)//2
    top=(INPUT_SIZE-nh)//2

    canvas[
        top:top+nh,
        left:left+nw
    ]=resized

    return canvas,scale,(left,top)

# --------------------------------
# NMS
# --------------------------------

def nms(boxes,scores,thr):

    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]

    area=(x2-x1)*(y2-y1)

    order=scores.argsort()[::-1]

    keep=[]

    while order.size:

        i=order[0]

        keep.append(i)

        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])

        inter=np.maximum(
            0,
            xx2-xx1
        ) * np.maximum(
            0,
            yy2-yy1
        )

        iou=inter/(
            area[i]+
            area[order[1:]]
            -inter+
            1e-6
        )

        order=order[
            np.where(
                iou<=thr
            )[0]+1
        ]

    return keep

# --------------------------------
# POSTPROCESS
# --------------------------------

def postprocess(outputs,shape,scale,pad):

    pred=outputs[0][0].T

    boxes=pred[:,:4]
    scores=pred[:,4:]

    conf=scores.max(axis=1)
    cls=scores.argmax(axis=1)

    mask=conf>=CONF

    boxes=boxes[mask]
    conf=conf[mask]
    cls=cls[mask]

    if len(conf)==0:
        return [],[],[]

    boxes[:,0]=(boxes[:,0]-pad[0])/scale
    boxes[:,1]=(boxes[:,1]-pad[1])/scale

    boxes[:,2:]/=scale

    boxes[:,0]-=boxes[:,2]/2
    boxes[:,1]-=boxes[:,3]/2

    boxes[:,2]+=boxes[:,0]
    boxes[:,3]+=boxes[:,1]

    keep=nms(
        boxes,
        conf,
        IOU
    )

    return (
        boxes[keep],
        conf[keep],
        cls[keep]
    )

# --------------------------------
# CAPTURE
# --------------------------------

def capture_loop():

    global latest_frame

    cap=cv2.VideoCapture(
        RTSP,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print("[RTSP] connected")

    while running:

        ret,frame=cap.read()

        if not ret:

            time.sleep(.02)
            continue

        with frame_lock:

            latest_frame=frame.copy()

# --------------------------------
# INFERENCE
# --------------------------------

def infer_loop():

    global latest_result

    rknn=RKNNLite()

    rknn.load_rknn(MODEL)

    rknn.init_runtime(
        core_mask=CORE
    )

    while running:

        with frame_lock:

            if latest_frame is None:
                frame=None
            else:
                frame=latest_frame.copy()

        if frame is None:

            time.sleep(.005)
            continue

        infer_frame=cv2.resize(
            frame,
            (960,540)
        )

        img,scale,pad=letterbox(
            infer_frame
        )

        img=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        outputs=rknn.inference(
            inputs=[
                np.expand_dims(
                    img,
                    0
                )
            ]
        )

        boxes,scores,cls=postprocess(
            outputs,
            infer_frame.shape,
            scale,
            pad
        )

        sx=frame.shape[1]/infer_frame.shape[1]
        sy=frame.shape[0]/infer_frame.shape[0]

        if len(boxes):

            boxes[:,0]*=sx
            boxes[:,1]*=sy
            boxes[:,2]*=sx
            boxes[:,3]*=sy

        with result_lock:

            latest_result=(
                frame,
                boxes,
                scores,
                cls
            )

# --------------------------------
# MAIN
# --------------------------------

def main():

    threading.Thread(
        target=capture_loop,
        daemon=True
    ).start()

    threading.Thread(
        target=infer_loop,
        daemon=True
    ).start()

    prev=time.time()

    while True:

        with result_lock:
            data=latest_result

        if data is None:

            time.sleep(.01)
            continue

        frame,boxes,scores,cls=data

        for b,s,c in zip(
            boxes,
            scores,
            cls
        ):

            x1,y1,x2,y2=map(
                int,
                b
            )

            cid=int(c)

            if cid<len(COCO_CLASSES):
                label=f"{COCO_CLASSES[cid]} {s:.2f}"
            else:
                label=f"class_{cid}"

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5,
                (0,255,0),
                2
            )

        now=time.time()

        fps=1/(now-prev+1e-6)
        prev=now

        cv2.putText(
            frame,
            f"FPS:{fps:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        h,w=frame.shape[:2]

        scale=min(
            1280/w,
            720/h
        )

        show=cv2.resize(
            frame,
            (
                int(w*scale),
                int(h*scale)
            )
        )

        cv2.imshow(
            "RKNN RTSP",
            show
        )

        if cv2.waitKey(1)==27:
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()