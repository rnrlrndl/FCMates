from ultralytics import YOLO
import torch
import datetime

def train():
    # YOLOv8n 모델
    model = YOLO("yolov8n.pt")

    # Train the Model
    results = model.train(
        data=r'C:\Users\win\FCMates\configs\data.yaml',
        epochs=100,
        patience=40,
        batch=8,
        imgsz=512,
        verbose=True,
        device='cuda',
        save=True,
        save_period=10,
        project=r'D:\results',
        name='exp01',
    )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()

