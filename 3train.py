"""
OLD FILE: D:\Light\step3_train.py
CHANGE: epochs and name
"""

from ultralytics import YOLO

print("Training starting...\n")

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=100,           
    imgsz=640,
    batch=8,
    name="headlight_v2",  
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    patience=20,
    save=True,
    plots=True,
    device="cpu"
)

print("""
Training Complete!
Model: runs/detect/headlight_v2/weights/best.pt
Next: python step4_detect.py
""")