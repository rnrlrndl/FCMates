import os, cv2, time, torch, yaml
import torchvision.ops as tv_ops
from ultralytics import YOLO

_orig_nms = tv_ops.nms

def patched_nms(boxes, scores, iou_threshold):
    idx = _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
    return idx.to(boxes.device)

tv_ops.nms = patched_nms

os.environ.setdefault(
    "LD_PRELOAD",
    "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0 "
    "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
)

PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1"
)

model = YOLO("tes.engine", task="detect")

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Could not open CSI camera")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer, record = None, False
frame_idx, fps = 0, 0.0
start = time.time()

print("Press  a  (rec),  f  (snapshot),  q  (quit)…")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed"); break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    ann   = model(frame, imgsz=640, verbose=False)[0].plot()

    frame_idx += 1
    if frame_idx % 20 == 0:
        fps = frame_idx / (time.time() - start)
    cv2.putText(ann, f"FPS:{fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLOv8 TensorRT", ann)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        record = not record
        if record:
            writer = cv2.VideoWriter("output.avi", fourcc, 30,
                                     (ann.shape[1], ann.shape[0]))
            print("REC on -> output.avi")
        else:
            if writer:
                writer.release(); writer = None
            print("REC off")

    elif key == ord('f'):
        fname = f"frame_{frame_idx}.jpg"
        cv2.imwrite(fname, ann)
        print(fname)

    elif key == ord('q'):
        break

    if record and writer:
        writer.write(ann)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("Finished")
