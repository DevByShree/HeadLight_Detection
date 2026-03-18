"""
headlight_demo/step4_detect.py

FINAL DETECTION:
- Detect every vehicle
- Check whether the headlight is legal or illegal
- Read the number plate of illegal vehicles
- Display a warning message
- Save a screenshot as evidence

Command: python step4_detect.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import json
import re

# OCR Setup
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    HAS_OCR = True
    print(" OCR ready")
except ImportError:
    HAS_OCR = False
    print("⚠ No OCR detected - pip install easyocr")
    print("  Plate detected but text not detected \n")


def read_plate(frame, bbox):
    if not HAS_OCR:
        return "OCR_NOT_INSTALLED"

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    plate_img = frame[y1:y2, x1:x2]
    if plate_img.size == 0:
        return "UNREADABLE"

    plate_img = cv2.resize(plate_img, None, fx=2, fy=2)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        results = reader.readtext(thresh, detail=0)
        if results:
            text = " ".join(results).upper().strip()
            text = re.sub(r'[^A-Z0-9 ]', '', text)
            return text if len(text) >= 4 else "UNREADABLE"
        return "UNREADABLE"
    except:
        return "UNREADABLE"


def find_nearest_plate(illegal_box, plates):
    ix = (illegal_box[0] + illegal_box[2]) / 2
    iy = (illegal_box[1] + illegal_box[3]) / 2

    nearest = None
    min_dist = float('inf')

    for p in plates:
        px = (p[0] + p[2]) / 2
        py = (p[1] + p[3]) / 2
        dist = np.sqrt((ix - px)**2 + (iy - py)**2)
        if dist < min_dist:
            min_dist = dist
            nearest = p
    return nearest


def process_video(model, video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output/{name}_result.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Video not opening: {video_path}")
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"\n🎬 {name} | Frames: {total}")

    violations = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model(frame, conf=0.15, verbose=False)

        legals = []
        illegals = []
        plates = []

        for result in results:
            for box in result.boxes:
                b = list(map(int, box.xyxy[0].tolist()))
                cls = model.names[int(box.cls[0])]
                if cls == "legal": legals.append(b)
                elif cls == "illegal": illegals.append(b)
                elif cls == "plate": plates.append(b)

        annotated = frame.copy()

        # Legal - GREEN
        for b in legals:
            cv2.rectangle(annotated, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)
            cv2.putText(annotated, "LEGAL", (b[0], b[1]-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Plates - YELLOW
        for b in plates:
            cv2.rectangle(annotated, (b[0],b[1]), (b[2],b[3]), (255,255,0), 2)

        # Illegal - RED + WARNING + PLATE READ
        for b in illegals:
            cv2.rectangle(annotated, (b[0],b[1]), (b[2],b[3]), (0,0,255), 4)
            cv2.putText(annotated, "ILLEGAL", (b[0], b[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            plate_text = "NOT DETECTED"
            nearest = find_nearest_plate(b, plates)

            if nearest:
                cv2.rectangle(annotated, (nearest[0],nearest[1]),
                            (nearest[2],nearest[3]), (0,0,255), 3)
                plate_text = read_plate(frame, nearest)

                cv2.rectangle(annotated, (nearest[0], nearest[3]),
                            (nearest[0]+len(plate_text)*16+10, nearest[3]+30),
                            (0,0,180), -1)
                cv2.putText(annotated, plate_text,
                          (nearest[0]+5, nearest[3]+22),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

                hc = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                pc = ((nearest[0]+nearest[2])//2, (nearest[1]+nearest[3])//2)
                cv2.line(annotated, hc, pc, (0,0,255), 2)

            # WARNING BOX
            time_str = datetime.now().strftime('%H:%M:%S')
            wx = w - 400
            wy = 80

            cv2.rectangle(annotated, (wx,wy), (w-10, wy+170), (0,0,150), -1)
            cv2.rectangle(annotated, (wx,wy), (w-10, wy+170), (0,0,255), 3)

            cv2.putText(annotated, "WARNING!", (wx+10, wy+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(annotated, "ILLEGAL HEADLIGHT", (wx+10, wy+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,255), 2)
            cv2.putText(annotated, f"Plate: {plate_text}", (wx+10, wy+90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
            cv2.putText(annotated, f"Time: {time_str}", (wx+10, wy+120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(annotated, "Fine: Rs.1000-5000", (wx+10, wy+150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            v = {'frame': frame_num, 'time': time_str,
                 'plate': plate_text, 'video': name}
            violations.append(v)

            ss_name = f"violation_{len(violations):04d}_{plate_text.replace(' ','_')}.jpg"
            cv2.imwrite(f"output/violations/{ss_name}", annotated)

        # Top bar
        if illegals:
            cv2.rectangle(annotated, (0,0), (w,60), (0,0,160), -1)
            cv2.putText(annotated,
                       f"ILLEGAL HEADLIGHT! | Violations: {len(violations)}",
                       (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.rectangle(annotated, (0,0), (w,40), (0,120,0), -1)
            cv2.putText(annotated, "ALL LEGAL", (15,28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out.write(annotated)

        show = cv2.resize(annotated, (1280, 720))
        cv2.imshow("Detection", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_num % 100 == 0:
            print(f"   {100*frame_num//max(total,1)}% | Violations: {len(violations)}")

    cap.release()
    out.release()
    print(f"    Done! Violations: {len(violations)}")
    return violations



if __name__ == "__main__":

    os.makedirs("output/violations", exist_ok=True)

    model_path = "runs/detect/headlight_V2/weights/best.pt"
    print(f"🔄 Loading: {model_path}")
    model = YOLO(model_path)
    print(" Model loaded!\n")

    videos = [
        "Videos/V1.webm",
        "Videos/V2.webm",
        "Videos/V3.webm",
        "Videos/V4.webm",
        "Videos/V5.webm",
    ]

    all_violations = []
    for video in videos:
        if os.path.exists(video):
            v = process_video(model, video)
            all_violations.extend(v)

    cv2.destroyAllWindows()

    # REPORT
    print("\n" + "█"*55)
    print("█  ILLEGAL HEADLIGHT DETECTION REPORT")
    print("█"*55)
    print(f"█  Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    print(f"█  Total Violations: {len(all_violations)}")
    print("█")

    if all_violations:
        print("█  #   | PLATE            | TIME     | VIDEO")
        print("█  " + "─"*50)
        for i, v in enumerate(all_violations[:20]):
            print(f"█  {i+1:3d} | {v['plate'][:16]:16s} | {v['time']:8s} | {v['video']}")
        print("█")
        print("█  Screenshots: output/violations/")
        print("█  Fine: Rs.1000-5000")
    else:
        print("█   NO VIOLATIONS FOUND!")

    print("█"*55)

    report = {
        'date': datetime.now().isoformat(),
        'total_violations': len(all_violations),
        'violations': all_violations
    }
    with open("output/violation_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n Report: output/violation_report.json")