from sympy.physics.quantum.circuitplot import pyplot
from ultralytics import YOLO
import matplotlib.pyplot as plt

def plot_metrics(metrics):
    labels = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    values = [
        metrics.box['precision'],
        metrics.box['recall'],
        metrics.box['mAP@0.5'],
        metrics.box['mAP@0.5:0.95']
    ]

    plt.bar(labels, values)
    plt.title("YOLOv8n Validation Results")
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('val_metrics_plot.png')
    plt.show()

def val():
    # Load Trained Model
    model = YOLO(r"D:\results\exp14\weights\best.pt")

    # Validate
    metrics = model.val(
        data=r'C:\Users\win\FCMates\configs\data.yaml',
        imgsz=416,
        batch=16,
        device='cuda',
        split='val',
        verbose=True
    )

    plot_metrics(metrics)
    print(metrics)

if __name__ == '__main__':
    val()