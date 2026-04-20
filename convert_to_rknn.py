from rknn.api import RKNN

ONNX_MODEL = 'yolov8s.onnx'
RKNN_MODEL = 'yolov8s.rknn'

# Create RKNN object
rknn = RKNN(verbose=True)

print('--> Config model')
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588'   # CHANGE if needed
)

print('--> Loading ONNX model')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('Load ONNX failed!')
    exit(ret)

print('--> Building RKNN model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build RKNN failed!')
    exit(ret)

print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export RKNN failed!')
    exit(ret)

print('RKNN model created successfully!')
