# Traffic Rule Violation Detection — AID 728

> **Course project** · Computer Vision · AID 728

---

## Overview

Modular pipeline that processes a single street image and returns a JSON dictionary describing every **two-wheeler violation** detected:

```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "DL 3C AB 1234"
    }
  ]
}
```

Only violating motorcycles appear in the output. Compliant vehicles are silently ignored.

---

## Pipeline (5 Stages)

```
Image
  ├─▶ Stage 1: YOLOv8n (COCO) ─── detect persons + motorcycles
  ├─▶ Stage 2: Geometric association ─── link riders to their motorcycle (IoA + proximity)
  ├─▶ Stage 3: Helmet classification ─── YOLOv8n helmet model + HSV heuristic fallback
  ├─▶ Stage 4: LP detection ─── YOLOv8n license-plate model
  └─▶ Stage 5: OCR ─── EasyOCR with CLAHE pre-processing
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights (internet required, one-time)

```bash
cd ..   # go to project root (parent of ROLL_NUMBER/)
python download_models.py
```

This downloads ~108 MB into `ROLL_NUMBER/models/`.

### 3. Quick test

```bash
python demo.py <path_to_image>          # print JSON
python demo.py <path_to_image> --show   # display annotated image
```

---

## Evaluation Interface

The evaluator imports the class and calls:

```python
from solution import TrafficViolationDetector
model = TrafficViolationDetector(model_dir="./models")
output = model.predict(image_path)
```

Both calls are handled correctly by this implementation.

---

## Model Sizes

| Model | Size |
|---|---|
| `yolov8n.pt` (COCO detector) | ~6 MB |
| `helmet_yolov8n.pt` | ~6 MB |
| `lp_detector.pt` | ~6 MB |
| EasyOCR English models | ~90 MB |
| **Total** | **~108 MB** ✓ |

---

## Directory Structure

```
ROLL_NUMBER/
├── solution.py          ← Main class (TrafficViolationDetector)
├── requirements.txt
├── README.md            ← This file
└── models/
    ├── yolov8n.pt
    ├── helmet_yolov8n.pt
    ├── lp_detector.pt
    └── easyocr/         ← EasyOCR model cache
```

---

## References

1. DashCop: Automated E-ticket Generation (arXiv 2503.00428)
2. Frontiers AI 2025 — YOLOv8 for Indian traffic violations
3. Sushaan Kanakaraj — Helmet violation detection thesis (NCIRL)
4. fast-plate-ocr (github.com/ankandrew/fast-plate-ocr)
5. YOLOv8 (Ultralytics)
