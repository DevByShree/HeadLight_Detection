"""
OLD FILE: D:\Light\step3_train.py
CHANGE: epochs and name
"""

from ultralytics import YOLO

print("Training starting...\n")

model = YOLO("yolov8s.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=150,           
    imgsz=800,
    batch=8,
    name="headlight_v3",  
    augment=True,
    hsv_h=0.015,
    hsv_s=0.8,
    hsv_v=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    scale=0.5,
    translate=0.1,
    lr0=0.01,
    lrf=0.001,
    warmup_epochs=5,
    patience=25,
    save=True,
    plots=True,
    device="cpu"
)

print("""
Training Complete!
Model: runs/detect/headlight_v2/weights/best.pt
Next: python step4_detect.py
""")