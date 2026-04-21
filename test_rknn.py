from rknnlite.api import RKNNLite

RKNN_MODEL = 'yolov8s.rknn'

rknn = RKNNLite()

print("Loading model...")
ret = rknn.load_rknn(RKNN_MODEL)
if ret != 0:
    print("Load failed")
    exit(ret)

print("Initializing runtime...")
ret = rknn.init_runtime()
if ret != 0:
    print("Runtime init failed")
    exit(ret)

print("Model loaded successfully!")
