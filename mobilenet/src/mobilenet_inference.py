import os
import torch
from torchvision import models, transforms
from PIL import Image
import json

# ✅ 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")
image_path = os.path.join(base_dir, "test", "test_image.png")  # ⬅ 여기만 변경됨
model_path = os.path.join(model_dir, "mobilenetv3-small-best.pth")
class_index_path = os.path.join(model_dir, "class_indices.json")

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ 이미지 로드 및 전처리
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

# ✅ 클래스 정보 로드
with open(class_index_path, encoding="utf-8") as f:
    class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

# ✅ 모델 정의 및 가중치 로드
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ✅ 추론
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()

# ✅ 출력
print("✅ 예측 결과:", idx_to_class[predicted_class])