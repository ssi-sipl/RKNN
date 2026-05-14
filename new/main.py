#!/usr/bin/env python3

import os
os.environ["OPENCV_VIDEOIO_DEBUG"]="0"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]=(
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay"
)

import cv2
import time
import threading
import argparse
import numpy as np
from rknnlite.api import RKNNLite

# ---------------------------------
# ARGS
# ---------------------------------

parser=argparse.ArgumentParser()

parser.add_argument("--rtsp",required=True)
parser.add_argument("--conf",type=float,default=.45)
parser.add_argument("--iou",type=float,default=.45)

args=parser.parse_args()

RTSP=args.rtsp
CONF=args.conf
IOU=args.iou

# ---------------------------------
# RKNN
# ---------------------------------

MODEL="yolov8s.rknn"

CORE=RKNNLite.NPU_CORE_0

INPUT=640

# ---------------------------------
# SHARED
# ---------------------------------

running=True

latest_frame=None
latest_result=None

frame_lock=threading.Lock()
result_lock=threading.Lock()

# ---------------------------------
# COCO
# ---------------------------------

COCO_CLASSES=[
"person","bicycle","car","motorcycle",
"bus","truck","dog","cat"
]

# ---------------------------------
# LETTERBOX
# ---------------------------------

def letterbox(img):

    h,w=img.shape[:2]

    scale=min(INPUT/w,INPUT/h)

    nw=int(w*scale)
    nh=int(h*scale)

    resized=cv2.resize(
        img,
        (nw,nh)
    )

    canvas=np.full(
        (INPUT,INPUT,3),
        114,
        dtype=np.uint8
    )

    left=(INPUT-nw)//2
    top=(INPUT-nh)//2

    canvas[
        top:top+nh,
        left:left+nw
    ]=resized

    return canvas,scale,(left,top)

# ---------------------------------
# NMS
# ---------------------------------

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

        xx1=np.maximum(
            x1[i],
            x1[order[1:]]
        )

        yy1=np.maximum(
            y1[i],
            y1[order[1:]]
        )

        xx2=np.minimum(
            x2[i],
            x2[order[1:]]
        )

        yy2=np.minimum(
            y2[i],
            y2[order[1:]]
        )

        inter=np.maximum(
            0,
            xx2-xx1
        )*np.maximum(
            0,
            yy2-yy1
        )

        iou=inter/(area[i]+area[order[1:]]-inter+1e-6)

        order=order[
            np.where(iou<=thr)[0]+1
        ]

    return keep

# ---------------------------------
# POST
# ---------------------------------

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

# ---------------------------------
# CAPTURE
# ---------------------------------

def capture_loop():

    global latest_frame
    global running

    cap=cv2.VideoCapture(
        RTSP,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        print("open failed")

        running=False

        return

    print("connected")

    while running:

        ret,frame=cap.read()

        if not ret:

            print("frame failed")

            time.sleep(.1)

            continue

        frame=cv2.resize(
            frame,
            (960,540)
        )

        with frame_lock:

            latest_frame=frame.copy()

    cap.release()

# ---------------------------------
# INFER
# ---------------------------------

def infer_loop():

    global latest_result

    rknn=RKNNLite()

    print("load")

    rknn.load_rknn(
        MODEL
    )

    print("runtime")

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

        img,scale,pad=letterbox(
            frame
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
            frame.shape,
            scale,
            pad
        )

        with result_lock:

            latest_result=(
                frame,
                boxes,
                scores,
                cls
            )

# ---------------------------------
# MAIN
# ---------------------------------

def main():

    t1=threading.Thread(
        target=capture_loop,
        daemon=True
    )

    t2=threading.Thread(
        target=infer_loop,
        daemon=True
    )

    t1.start()
    t2.start()

    prev=time.time()

    while running:

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

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"{COCO_CLASSES[int(c)]} {s:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                .6,
                (0,255,0),
                2
            )

        now=time.time()

        fps=1/(now-prev)

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

        cv2.imshow(
            "RKNN",
            frame
        )

        if cv2.waitKey(1)==27:
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()