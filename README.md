# 🚨 Illegal Headlight Detection System

> An AI-powered traffic enforcement tool that detects illegal vehicle headlights, reads number plates, and auto-generates violation evidence using YOLOv8 and OCR.

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run-step-by-step)
  - [Step 1 — Extract Frames](#step-1--extract-frames-from-videos)
  - [Step 2 — Label Frames](#step-2--label-the-frames)
  - [Step 3 — Train Model](#step-3--train-the-ai-model)
  - [Step 4 — Run Detection](#step-4--run-detection-on-videos)
- [Detection Classes](#-detection-classes)
- [Sample Output](#-sample-output)
- [Common Problems & Solutions](#-common-problems--solutions)
- [Tips for Better Accuracy](#-tips-for-better-accuracy)
- [Fine Information](#-fine-information-india--motor-vehicles-act)
- [Built With](#-built-with)

---

## ✅ What This Project Does

- 📹 Takes traffic/road video as input
- 🔍 Detects whether a vehicle has a **legal** or **illegal** headlight
- 🔤 Reads the **number plate** of illegal vehicles using OCR
- ⚠️ Shows a **warning message** on screen with fine details
- 📸 Saves a **screenshot** of each violation as evidence
- 📄 Generates a **violation report** in JSON format

---

## 📁 Project Structure

```
Light/
│
├── Videos/                      # Your input videos
│   ├── V1.webm
│   ├── V2.webm
│   ├── V3.webm
│   ├── V4.webm
│   └── V5.webm
│
├── 1extract.py                  # Step 1 — Extract frames
├── 2label.py                    # Step 2 — Label frames
├── 3train.py                    # Step 3 — Train model
├── 4detect.py                   # Step 4 — Run detection
├── README.md
│
├── all_frames/                  # Extracted frames (auto-created)
│   ├── V1_f00000.jpg
│   └── V1_f00010.jpg
│
├── dataset/                     # Labeled data (auto-created)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
├── runs/                        # Trained model (auto-created)
│   └── detect/
│       └── headlight_v3/
│           └── weights/
│               └── best.pt      # ← Your final model
│
└── output/                      # Results (auto-created)
    ├── V1_result.mp4
    ├── violations/              # Violation screenshots
    └── violation_report.json
```

---

## 🔧 Requirements

### Python Version
- Python **3.9 or above** → [Download Here](https://www.python.org/downloads/)

### Install Required Libraries

Open **Command Prompt / Terminal** and run:

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install easyocr
```

> **Note:** `easyocr` is optional. Without it, plates will be detected but the text on them will not be read.

---

## 🚀 How to Run — Step by Step

### Step 1 — Extract Frames from Videos

Reads each video from the `Videos/` folder and saves every 10th frame as a `.jpg` image into `all_frames/`.

```bash
python 1extract.py
```

> 💡 You can change `frame_gap = 10` inside the file to extract frames more or less frequently.

---

### Step 2 — Label the Frames

Opens a window where you draw bounding boxes around headlights and number plates using your mouse.

```bash
python 2label.py
```

#### Controls

| Action | Key / Mouse |
|--------|-------------|
| Draw a box | Click + Drag |
| Mark as **Legal** headlight | Press `L` |
| Mark as **Illegal** headlight | Press `I` |
| Mark as **Number Plate** | Press `P` |
| Undo last box | Press `U` |
| Save & go to next frame | Press `N` |
| Skip frame (don't save) | Press `S` |
| Quit | Press `Q` |

#### How to Label Each Vehicle

1. Draw a box around the headlight → press `L` (legal) or `I` (illegal)
2. Draw a box around the number plate → press `P`
3. Repeat for all vehicles in the frame
4. Press `N` to save and move to the next frame

**Output →** `dataset/` folder with images, labels, and `data.yaml`

---

### Step 3 — Train the AI Model

Trains YOLOv8 to recognize legal headlights, illegal headlights, and number plates.

```bash
python 3train.py
```

#### Training Settings

| Setting | Default | What It Does |
|---------|---------|--------------|
| `epochs` | `150` | How many times the model learns the data |
| `imgsz` | `800` | Image size used during training |
| `batch` | `8` | Number of images processed at once |
| `device` | `cpu` | Use `0` for NVIDIA GPU, `cpu` for CPU |
| `patience` | `25` | Stops early if no improvement |

#### Estimated Training Time

| Device | Time |
|--------|------|
| CPU | 2–6 hours |
| GPU (NVIDIA) | 15–45 minutes |

> 💡 **Tip:** If you have an NVIDIA GPU, change `device="cpu"` to `device=0` in `3train.py` for much faster training.

**Output →** Trained model saved at `runs/detect/headlight_v3/weights/best.pt`

---

### Step 4 — Run Detection on Videos

Runs your trained model on the input videos and identifies all violations.

```bash
python 4detect.py
```

> ⚠️ Before running, verify the model path inside `4detect.py`:
> ```python
> model_path = "runs/detect/headlight_v3/weights/best.pt"
> ```

#### What It Does

- Goes through each video frame by frame
- Draws 🟢 **GREEN** box on legal headlights
- Draws 🔴 **RED** box on illegal headlights
- Reads the number plate of illegal vehicles
- Shows a **WARNING** box with plate number and fine amount
- Saves a screenshot of every violation
- Creates an output video with all detections marked
- Generates a JSON violation report

Press `Q` anytime to stop detection.

#### Output Files

| File | Description |
|------|-------------|
| `output/V1_result.mp4` | Video with detections drawn |
| `output/violations/` | Screenshot of each violation |
| `output/violation_report.json` | Complete violation report |

---

## 🏷️ Detection Classes

| Class ID | Name | Box Color | Meaning |
|----------|------|-----------|---------|
| `0` | `legal` | 🟢 Green | Normal / allowed headlight |
| `1` | `illegal` | 🔴 Red | Modified / illegal headlight |
| `2` | `plate` | 🟡 Yellow | Vehicle number plate |

---

## 🖥️ Sample Output

#### Detection Window Preview

```
┌──────────────────────────────────────────────┐
│  ⚠ ILLEGAL HEADLIGHT!  |  Violations: 3      │
│                                              │
│    ┌────────┐                 ┌────────────┐ │
│    │ILLEGAL │  (red box)      │  WARNING!  │ │
│    └────────┘                 │  ILLEGAL   │ │
│    ┌────────┐                 │  Plate: XX │ │
│    │MH12AB  │  (plate box)    │  Fine:1000 │ │
│    └────────┘                 └────────────┘ │
└──────────────────────────────────────────────┘
```

#### Violation Report (`violation_report.json`)

```json
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
```

---

## 🛠️ Common Problems & Solutions

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| `ModuleNotFoundError: easyocr` | Run `pip install easyocr` |
| Video file does not exist | Check the video file name and path |
| Model not found | Check `model_path` in `4detect.py` |
| Training is very slow | Use GPU (`device=0`) or reduce `epochs` |
| Low detection accuracy | Label more frames (200+) or increase `epochs` |
| `warmup_ephochs` error | Fix typo → change to `warmup_epochs` |
| Out of memory error | Reduce batch size to `4` or `2` |
| EasyOCR not reading plates | Ensure the plate is visible and not blurry |

---

## 💡 Tips for Better Accuracy

- 📦 **Label 200+ frames** — more data = better results
- 🚗 **Label ALL vehicles** in every frame — don't skip any
- 📐 **Include different angles** — front view, slight side, far, close
- 🌙 **Include day AND night videos** — the model needs to learn both
- ✂️ **Draw tight boxes** — minimize background around the object
- ⚖️ **Balance your classes** — aim for roughly equal legal and illegal label counts

---

## 🇮🇳 Fine Information (India — Motor Vehicles Act)

| Violation | Fine |
|-----------|------|
| Illegal headlight modification | ₹1,000 – ₹5,000 |
| Section 177 MV Act | General traffic violation |
| Section 190(2) MV Act | Vehicle alteration without permit |

---

## 🛠️ Built With

| Tool / Library | Purpose |
|----------------|---------|
| Python 3.11 | Programming language |
| YOLOv8 (Ultralytics) | Object detection AI model |
| OpenCV | Video & image processing |
| EasyOCR | Number plate text reading |
| NumPy | Array & math operations |

---
