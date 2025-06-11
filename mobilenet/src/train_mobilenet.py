import os
import json
import torch
import sys
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from torch.cuda.amp import GradScaler, autocast

def train():
    # 경로 설정 (프로젝트 루트 기준)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    transform_train = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    num_classes = len(train_dataset.classes)

    with open(os.path.join(model_dir, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[2].p = 0.2
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler()

    best_accuracy = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                sys.stdout.write(f"\r[Epoch {epoch+1}] ▶ Batch {batch_idx+1}/{len(train_loader)} 진행 중...")
                sys.stdout.flush()

        print()
        avg_loss = running_loss / len(train_loader)

        # 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        scheduler.step()
        elapsed_time = time.time() - start_time
        print(f"[{epoch+1}/{num_epochs}] 🔧 Loss: {avg_loss:.4f} | ✅ Val Accuracy: {accuracy:.2f}% | ⏱ Time: {elapsed_time:.1f}s")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(model_dir, "mobilenetv3-small-best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"📌 Best model updated and saved at epoch {epoch+1}")

    print("✅ 학습 완료")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()