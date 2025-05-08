from ultralytics import YOLO
import torch
import datetime

def train():
    # YOLOv8n 모델
    model = YOLO("yolov8n.pt")

    # Train the Model
    results = model.train(
        data=r'C:\Users\win\FCMates\configs\data.yaml',
        epochs=20,
        patience=20,
        batch=8,
        imgsz=512,
        lr0=0.005,      # 초기 값: 0.01
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.937,
        box=0.1,        # 초기 값 : 0.05
        cls=0.5,
        hsv_h=0.015,
        hsv_s=0.5,      # 초기 값 : 0.7
        hsv_v=0.4,
        mosaic=0.7,     # 초기 값 : 1.0
        copy_paste=0.1, #  초기 값 : 0.0
        cos_lr=True,
        optimizer='AdamW',
        verbose=True,
        device='cuda',
        save=True,
        save_period=10,
        project=r'D:\results',
        name='exp01',
        workers=4
    )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()

