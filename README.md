#  Illegal Headlight Detection System

This project detects **illegal headlights** on vehicles using AI (YOLOv8), reads the **number plate** of violators, and saves evidence automatically.

---

##  What This Project Does

- Takes traffic/road video as input
- Detects whether a vehicle has **legal** or **illegal** headlight
- Reads the **number plate** of illegal vehicles using OCR
- Shows **warning message** on screen with fine details
- Saves **screenshot** of each violation as evidence
- Generates a **violation report** in JSON format

---

Light/
|
|-- Videos/ (Your input videos)
| |-- V1.webm
| |-- V2.webm
| |-- V3.webm
| |-- V4.webm
| |-- V5.webm
|
|-- 1extract.py (Step 1 - Extract frames)
|-- 2label.py (Step 2 - Label frames)
|-- 3train.py (Step 3 - Train model)
|-- 4detect.py (Step 4 - Run detection)
|-- README.md
|
|-- all_frames/ (Extracted frames - auto created)
| |-- V1_f00000.jpg
| |-- V1_f00010.jpg
|
|-- dataset/ (Labeled data - auto created)
| |-- images/
| | |-- train/
| | |-- val/
| |-- labels/
| | |-- train/
| | |-- val/
| |-- data.yaml
|
|-- runs/ (Trained model - auto created)
| |-- detect/
| |-- headlight_v3/
| |-- weights/
| |-- best.pt (Your final model)
|
|-- output/ (Results - auto created)
|-- V1_result.mp4 (Output video)
|-- violations/ (Violation screenshots)
|-- violation_report.json (Final report)


---

##  Requirements

### Python Version
- Python **3.9 or above** → [Download Here](https://www.python.org/downloads/)

### Install Required Libraries

Open **Command Prompt / Terminal** and run:

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install easyocr
 easyocr is optional. Without it, plates will be detected
but the text on the plate will not be read.

 How to Run — Step by Step
Step 1 — Extract Frames from Videos
This takes your videos and saves individual frames (images) from them.

Bash

python 1extract.py
What happens:

Reads each video from Videos/ folder
Saves every 10th frame as a .jpg image
All frames are saved in all_frames/ folder
You can change frame_gap = 10 to any other number
Output → all_frames/ folder with .jpg images

Step 2 — Label the Frames
This opens a window where you draw boxes on headlights and number plates using your mouse.

Bash

python 2label.py
Controls:

Action	Key / Mouse
Draw a box	Click + Drag with mouse
Mark as Legal headlight	Press l
Mark as Illegal headlight	Press i
Mark as Number Plate	Press p
Undo last box	Press u
Save & go to next frame	Press n
Skip frame (don't save)	Press s
Quit	Press q
How to label each vehicle:

Draw a box around the headlight → press l (legal) or i (illegal)
Draw a box around the number plate → press p
Repeat for all vehicles in the frame
Press n to save and move to the next frame
Output → dataset/ folder with images, labels, and data.yaml

Step 3 — Train the AI Model
This trains YOLOv8 to recognize legal headlights, illegal headlights, and number plates.

Bash

python 3train.py
Settings you can change inside the file:

Setting	Default	What It Does
epochs	150	How many times model learns the data
imgsz	800	Image size used during training
batch	8	Number of images processed at once
device	cpu	Use 0 for NVIDIA GPU, cpu for CPU
patience	25	Stops early if no improvement
Estimated Training Time:

Device	Time
CPU	2–6 hours
GPU	15–45 minutes
 If you have an NVIDIA GPU, change device="cpu" to device=0
in 3train.py for much faster training.

Output → Trained model saved at runs/detect/headlight_v3/weights/best.pt

Step 4 — Run Detection on Videos
This runs your trained model on the videos and finds all violations.

Bash

python 4detect.py
 Before running, check the model path inside 4detect.py:

Python

model_path = "runs/detect/headlight_v3/weights/best.pt"
Change it if your model is saved at a different location.

What it does:

Goes through each video frame by frame
Draws 🟢 GREEN box on legal headlights
Draws 🔴 RED box on illegal headlights
Reads the number plate of illegal vehicles
Shows a WARNING box with plate number and fine amount
Saves a screenshot of every violation
Creates an output video with all detections marked
Generates a JSON report of all violations
Press q anytime to stop detection.

Output:

output/V1_result.mp4 → Video with detections drawn
output/violations/ → Screenshot of each violation
output/violation_report.json → Complete violation report
 Detection Classes
Class ID	Name	Box Color	Meaning
0	legal	🟢 Green	Normal / allowed headlight
1	illegal	🔴 Red	Modified / illegal headlight
2	plate	🟡 Yellow	Vehicle number plate
Sample Output
Detection Window Preview
text

┌──────────────────────────────────────────────┐
│  ⚠ ILLEGAL HEADLIGHT!  |  Violations: 3     │
│                                              │
│    ┌────────┐                 ┌────────────┐ │
│    │ILLEGAL │ (red box)       │  WARNING!  │ │
│    └────────┘                 │  ILLEGAL   │ │
│    ┌────────┐                 │  Plate: XX │ │
│    │MH12AB  │ (plate box)     │  Fine:1000 │ │
│    └────────┘                 └────────────┘ │
└──────────────────────────────────────────────┘
Violation Report (JSON)
JSON

{
  "date": "2025-01-15T20:30:00",
  "total_violations": 5,
  "violations": [
    {
      "frame": 120,
      "time": "20:30:15",
      "plate": "MH12AB1234",
      "video": "V1"
    }
  ]
}
 Common Problems & Solutions
Problem	Solution
ModuleNotFoundError: ultralytics	Run pip install ultralytics
ModuleNotFoundError: easyocr	Run pip install easyocr
Video file does not exist	Check video file name and path
Model not found	Check model_path in 4detect.py
Training is very slow	Use GPU (device=0) or reduce epochs
Low detection accuracy	Label more frames (200+) or increase epochs
warmup_ephochs error	Fix typo → change to warmup_epochs
Out of memory error	Reduce batch size to 4 or 2
EasyOCR not reading plates properly	Ensure plate is visible and not blurry
 Tips for Better Accuracy
Label at least 200+ frames — more data = better results
Label ALL vehicles in every frame — don't skip any
Include different angles — front view, slight side, far, close
Include day AND night videos — model needs to learn both
Draw tight boxes — don't include too much background around the object
Balance your classes — try to have roughly equal numbers of legal and illegal labels
 Fine Information (India — Motor Vehicles Act)
Violation	Fine
Illegal headlight modification	₹1,000 – ₹5,000
Section 177 MV Act	General traffic violation
Section 190(2) MV Act	Vehicle alteration without permit
 Built With
Tool / Library	Purpose
Python 3.11	Programming language
YOLOv8 (Ultralytics)	Object detection AI model
OpenCV	Video & image processing
EasyOCR	Number plate text reading
NumPy	Array & math operations
Author
Shree Joshi 
College Project — Illegal Headlight Detection System
