from ultralytics import YOLO

print("Training starting...\n")

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="headlight_plate",
    augment=True,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.5,
    fliplr=0.5,
    mosaic=1.0,
    patience=15,
    save=True,
    plots=True,
    device="cpu"
)

print("""
╔═══════════════════════════════════════╗
║   Training Complete!                ║
║                                       ║
║  Model: runs/detect/headlight_plate/  ║
║         weights/best.pt               ║
║                                       ║
║  Next: python step4_detect.py         ║
╚═══════════════════════════════════════╝
""")