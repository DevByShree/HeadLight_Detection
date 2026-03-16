import cv2
import os
import shutil
import random


class Labeler:

    def __init__(self):
        self.frames_dir = "all_frames"
        self.output_dir = "dataset"

        # Folders check
        for split in ['train', 'val']:
            os.makedirs(f"{self.output_dir}/images/{split}", exist_ok=True)
            os.makedirs(f"{self.output_dir}/labels/{split}", exist_ok=True)

        # Frames list
        self.frames = sorted([
            f for f in os.listdir(self.frames_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

        # Mouse drawing state
        self.drawing = False
        self.sx = 0    # start x
        self.sy = 0    # start y
        self.ex = 0    # end x
        self.ey = 0    # end y

        # Current frame data
        self.boxes = []           #  boxes
        self.original = None      # original frame
        self.display = None       # display frame
        self.waiting = False      # label waiting?
        self.idx = 0              # current frame index

        # Stats
        self.legal_count = 0
        self.illegal_count = 0
        self.plate_count = 0
        self.frame_count = 0

        # Colors
        self.colors = {
            'legal':   (0, 255, 0),     # GREEN
            'illegal': (0, 0, 255),     # RED
            'plate':   (255, 255, 0),   # CYAN/YELLOW
        }

    def mouse_event(self, event, x, y, flags, param):
        """Mouse se box draw karna"""

        # ── Mouse click start ──
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.sx = x
            self.sy = y
            self.ex = x
            self.ey = y

        # ── Mouse drag ──
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.ex = x
                self.ey = y
                # Live rectangle 
                self.refresh()
                cv2.rectangle(
                    self.display,
                    (self.sx, self.sy),
                    (self.ex, self.ey),
                    (0, 255, 255), 2   # Yellow while drawing
                )
                cv2.imshow("Label Tool", self.display)

        # ── Mouse release ──
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.ex = x
            self.ey = y

            # Box ka size check
            w = abs(self.ex - self.sx)
            h = abs(self.ey - self.sy)

            if w > 15 and h > 15:
                # Valid box - ab label maang
                self.waiting = True
                self.refresh()

                x1 = min(self.sx, self.ex)
                y1 = min(self.sy, self.ey)
                x2 = max(self.sx, self.ex)
                y2 = max(self.sy, self.ey)

                # Pending box dikha (yellow)
                cv2.rectangle(self.display, (x1, y1), (x2, y2),
                              (0, 255, 255), 3)

                # Instruction
                cv2.rectangle(self.display, (x1, y2),
                              (x1 + 380, y2 + 35), (0, 0, 0), -1)
                cv2.putText(self.display,
                            "Press: 'l'=Legal  'i'=Illegal  'p'=Plate",
                            (x1 + 5, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)

                cv2.imshow("Label Tool", self.display)
            else:
                print("   ⚠ Box bahut chhota hai, phir se try kar")

    def refresh(self):
        """Frame + existing boxes redraw kar"""
        self.display = self.original.copy()

        # Existing boxes draw kar
        for box in self.boxes:
            bx1, by1, bx2, by2 = box['bbox']
            label = box['label']
            color = self.colors[label]

            cv2.rectangle(self.display, (bx1, by1), (bx2, by2), color, 3)

            text = label.upper()
            cv2.rectangle(self.display, (bx1, by1 - 28),
                          (bx1 + len(text) * 16 + 10, by1), color, -1)
            cv2.putText(self.display, text, (bx1 + 5, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Top info bar
        h, w = self.display.shape[:2]
        cv2.rectangle(self.display, (0, 0), (w, 70), (40, 40, 40), -1)

        line1 = (f"Frame: {self.idx + 1}/{len(self.frames)} | "
                 f"Boxes: {len(self.boxes)} | "
                 f"Legal:{self.legal_count} "
                 f"Illegal:{self.illegal_count} "
                 f"Plates:{self.plate_count}")

        line2 = ("Draw box with mouse | "
                 "l=Legal i=Illegal p=Plate | "
                 "u=Undo n=Next s=Skip q=Quit")

        cv2.putText(self.display, line1, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(self.display, line2, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def save_labels(self, frame_path):
        """YOLO format mein save kar"""

        # 80% train, 20% val
        split = 'train' if random.random() < 0.8 else 'val'

        fname = os.path.basename(frame_path)
        name = os.path.splitext(fname)[0]

        # Image copy
        shutil.copy2(frame_path,
                     f"{self.output_dir}/images/{split}/{fname}")

        # Label file bana
        img = cv2.imread(frame_path)
        ih, iw = img.shape[:2]

        label_path = f"{self.output_dir}/labels/{split}/{name}.txt"

        with open(label_path, 'w') as f:
            for box in self.boxes:
                bx1, by1, bx2, by2 = box['bbox']
                label = box['label']

                # Class ID
                class_map = {'legal': 0, 'illegal': 1, 'plate': 2}
                cls_id = class_map[label]

                # YOLO format: normalized center x, y, width, height
                xc = ((bx1 + bx2) / 2) / iw
                yc = ((by1 + by2) / 2) / ih
                bw = abs(bx2 - bx1) / iw
                bh = abs(by2 - by1) / ih

                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def create_yaml(self):
        """data.yaml file bana"""

        abs_path = os.path.abspath(self.output_dir)

        yaml_text = f"""path: {abs_path}
train: images/train
val: images/val

nc: 3
names:
  - legal
  - illegal
  - plate
"""
        with open(f"{self.output_dir}/data.yaml", 'w') as f:
            f.write(yaml_text)

    def run(self):
        """Main labeling loop"""

        print(f"""
╔═══════════════════════════════════════════════════╗
║         HEADLIGHT + PLATE LABELING TOOL           ║
╠═══════════════════════════════════════════════════╣
║  Frames: {len(self.frames)}
║                                                   
║  MOUSE SE BOX DRAW KAR, PHIR KEY PRESS KAR:      
║                                                   
║  'l' = Legal headlight    (GREEN box)             
║  'i' = Illegal headlight  (RED box)               
║  'p' = Number plate       (YELLOW box)            
║                                                   
║  HAR GAADI KE LIYE:                               
║  → Headlight pe box draw → 'l' ya 'i'            
║  → Plate pe box draw → 'p'                        
║                                                   
║  'u' = Undo last box                              
║  'n' = Next frame (save + agle pe jaa)            
║  's' = Skip frame                                 
║  'q' = Quit                                       
╚═══════════════════════════════════════════════════╝
        """)

        cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Label Tool", 1280, 720)
        cv2.setMouseCallback("Label Tool", self.mouse_event)

        while self.idx < len(self.frames):
            fname = self.frames[self.idx]
            fpath = os.path.join(self.frames_dir, fname)

            # Frame load
            self.original = cv2.imread(fpath)
            if self.original is None:
                self.idx += 1
                continue

            # Resize agar bahut bada hai
            h, w = self.original.shape[:2]
            if w > 1920:
                scale = 1920 / w
                self.original = cv2.resize(
                    self.original, (int(w * scale), int(h * scale))
                )

            self.boxes = []
            self.waiting = False
            self.refresh()
            cv2.imshow("Label Tool", self.display)

            # Key press wait loop
            while True:
                key = cv2.waitKey(30) & 0xFF

                # ── LEGAL ──
                if key == ord('l') and self.waiting:
                    x1 = min(self.sx, self.ex)
                    y1 = min(self.sy, self.ey)
                    x2 = max(self.sx, self.ex)
                    y2 = max(self.sy, self.ey)

                    self.boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': 'legal'
                    })
                    self.legal_count += 1
                    self.waiting = False
                    print(f"   ✅ LEGAL headlight added")

                    self.refresh()
                    cv2.imshow("Label Tool", self.display)

                # ── ILLEGAL ──
                elif key == ord('i') and self.waiting:
                    x1 = min(self.sx, self.ex)
                    y1 = min(self.sy, self.ey)
                    x2 = max(self.sx, self.ex)
                    y2 = max(self.sy, self.ey)

                    self.boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': 'illegal'
                    })
                    self.illegal_count += 1
                    self.waiting = False
                    print(f"   ❌ ILLEGAL headlight added")

                    self.refresh()
                    cv2.imshow("Label Tool", self.display)

                # ── PLATE ──
                elif key == ord('p') and self.waiting:
                    x1 = min(self.sx, self.ex)
                    y1 = min(self.sy, self.ey)
                    x2 = max(self.sx, self.ex)
                    y2 = max(self.sy, self.ey)

                    self.boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': 'plate'
                    })
                    self.plate_count += 1
                    self.waiting = False
                    print(f"   🔤 NUMBER PLATE added")

                    self.refresh()
                    cv2.imshow("Label Tool", self.display)

                # ── UNDO ──
                elif key == ord('u'):
                    if self.boxes:
                        removed = self.boxes.pop()
                        lbl = removed['label']
                        if lbl == 'legal':
                            self.legal_count -= 1
                        elif lbl == 'illegal':
                            self.illegal_count -= 1
                        else:
                            self.plate_count -= 1
                        print(f"   ↩ Undo: {lbl} removed")
                        self.refresh()
                        cv2.imshow("Label Tool", self.display)

                # ── NEXT FRAME ──
                elif key == ord('n'):
                    if self.boxes:
                        self.save_labels(fpath)
                        self.frame_count += 1
                        print(f"   💾 Saved {len(self.boxes)} boxes "
                              f"(Frame {self.frame_count})")
                    else:
                        print(f"   ⏭ No boxes, skipped")
                    self.idx += 1
                    break

                # ── SKIP ──
                elif key == ord('s'):
                    print(f"   ⏭ Skipped")
                    self.idx += 1
                    break

                # ── QUIT ──
                elif key == ord('q'):
                    if self.boxes:
                        self.save_labels(fpath)
                        self.frame_count += 1
                    self.idx = len(self.frames)
                    break

        cv2.destroyAllWindows()

        # data.yaml bana
        self.create_yaml()

        # Summary
        train_imgs = len(os.listdir(f"{self.output_dir}/images/train"))
        val_imgs = len(os.listdir(f"{self.output_dir}/images/val"))

        print(f"""
╔═══════════════════════════════════════════════╗
║           ✅ LABELING COMPLETE!               ║
╠═══════════════════════════════════════════════╣
║                                               
║  📊 Stats:                                    
║     Frames labeled: {self.frame_count}
║     Legal boxes:    {self.legal_count}
║     Illegal boxes:  {self.illegal_count}
║     Plate boxes:    {self.plate_count}
║     Total boxes:    {self.legal_count + self.illegal_count + self.plate_count}
║                                               
║  📁 Dataset:                                  
║     Train: {train_imgs} images
║     Val:   {val_imgs} images
║     YAML:  dataset/data.yaml
║                                               
║  ✅ Next: python step3_train.py               
╚═══════════════════════════════════════════════╝
        """)


# ═══ RUN ═══
if __name__ == "__main__":
    labeler = Labeler()
    labeler.run()