from ultralytics import YOLO

def predict():
    model = YOLO('results/exp1/weights/best.pt')

    results = model.predict(
        source='',  # 영상
        imgsz=416,
        conf=0.25,
        save=True,
        save_txt=True,
        device='cuda',
        name='predict_exp1',
        verbose=True
    )

if __name__ == '__main__':
    predict()