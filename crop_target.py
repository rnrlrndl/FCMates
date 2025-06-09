import os
import cv2
import json
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

# -------------------- 경로 설정 --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# MobileNet 관련 경로
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv3-small-best.pth")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# -------------------- 모델 로딩 --------------------
# YOLOv8 모델 (사전 학습된)
yolo_model = YOLO("yolov8n.pt")

# UNet 모델 (예시: 사용자 정의로 교체)
class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=1.0)

    def forward(self, x):
        return self.up(x)

unet_model = DummyUNet()
unet_model.eval()

# MobileNetV3 모델 로드
with open(CLASS_INDEX_PATH, encoding="utf-8") as f:
    class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

mobilenet_model = models.mobilenet_v3_small(weights=None)
mobilenet_model.classifier[3] = nn.Linear(mobilenet_model.classifier[3].in_features, num_classes)
mobilenet_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
mobilenet_model.eval()

# -------------------- 전처리 설정 --------------------
resize_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor()
])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# -------------------- 실시간 처리 --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

CONF_THRESH = 0.5
SKIP_FRAME = 2
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAME != 0:
        cv2.imshow("Traffic Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO 추론
    results = yolo_model(frame)[0]
    boxes = results.boxes.xyxy.tolist()
    confs = results.boxes.conf.tolist()
    classes = results.boxes.cls.tolist()

    for i, box in enumerate(boxes):
        score = confs[i]
        class_id = int(classes[i])

        if score < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor_crop = resize_transform(pil_crop).unsqueeze(0)

        if class_id == 0:  # 손상된 표지판
            with torch.no_grad():
                restored = unet_model(tensor_crop)
        else:
            restored = tensor_crop  # 정상인 경우 그대로

        # MobileNet 추론
        with torch.no_grad():
            input_tensor = normalize(restored.squeeze(0)).unsqueeze(0)
            output = mobilenet_model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = idx_to_class[pred_idx]

        # 결과 출력
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{pred_label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
