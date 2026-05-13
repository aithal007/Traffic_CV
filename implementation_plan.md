# Traffic Rule Violation Detection — AID 728
## Implementation Plan (v2 — informed by all 9 reference papers)

---

## Overview

Build a **modular 5-stage pipeline** that processes a single street image and returns a JSON dict of violations, staying well within the 250 MB model limit. The design is inspired directly by the DashCop (arxiv 2503.00428) approach, simplified for single-image (no video/tracking) use.

---

## Architecture

```
Image
  │
  ▼
Stage 1: Two-Wheeler Detection      ← YOLOv8n (COCO), classes: motorcycle(3), bicycle(1)
  │
  ▼
Stage 2: Person Detection           ← YOLOv8n (same model), class: person(0)
  │
  ▼
Stage 3: Rider–Motorcycle Association  ← geometric IoA + vertical proximity scoring
  │
  ▼
Stage 4: Helmet Classification      ← YOLOv8n fine-tuned on helmet/no-helmet
                                      (fallback: head-region HSV heuristic)
  │
  ▼
Stage 5: License Plate Detection   ← YOLOv8n fine-tuned on LP (license-plate-yolov8n)
         + OCR                     ← EasyOCR (en) with CLAHE preprocessing
  │
  ▼
Output JSON (violations only)
```

---

## Model Budget

| Model | File | Est. Size |
|---|---|---|
| YOLOv8n (COCO) — detection | `yolov8n.pt` | ~6 MB |
| YOLOv8n helmet detector | `helmet_yolov8n.pt` | ~6 MB |
| License plate detector | `lp_detector.pt` | ~6 MB |
| EasyOCR (English) | cached in `models/easyocr/` | ~90 MB |
| **Total** | | **~108 MB** ✅ |

---

## Stage-by-Stage Design

### Stage 1+2: Detection (YOLOv8n COCO)
- Single inference pass detects persons (class 0), motorcycles (class 3)
- conf_thresh=0.25 for persons, 0.30 for motorcycles
- NMS applied automatically by ultralytics

### Stage 3: Association (DashCop-inspired, bounding-box level)
Based on the IoA (Intersection over Area) approach from the literature:
- For each motorcycle bbox M, find all person bboxes P where:
  - `IoA(P, M) = area(P∩M) / area(P)` > 0.15 (person partially overlaps motorcycle)
  - OR centroid of P lies within 1.5× height of M above M's top edge (rider sitting above)
  - OR horizontal center of P is within M's horizontal span
- Score = IoA + vertical_proximity_bonus
- Hungarian matching for optimal assignment (scipy.optimize.linear_sum_assignment)
- A person not assigned to any motorcycle = pedestrian, skip

### Stage 4: Helmet Detection
**Primary**: YOLOv8n trained on helmet/no-helmet dataset
- Run on upper-body crop of each rider (top 40% of person bbox)
- Classes: helmet (0), no_helmet (1)
- conf_thresh = 0.35

**Fallback heuristic** (if model not available / low confidence):
- Crop head region (top 20% of person bbox)
- Convert to HSV
- Check variance of saturation in head region
- Low variance + rounded blob → helmet
- High skin-tone pixel ratio → no helmet

### Stage 5: License Plate OCR
**Detection**: Dedicated LP detector (YOLOv8n fine-tuned)
- Search in motorcycle bbox region (expanded by 30%)
- Also search full image with lower threshold

**OCR chain** (try in order, return first confident result):
1. EasyOCR (en) on CLAHE-preprocessed crop
2. Tesseract fallback (if installed)
3. Character-level heuristics (return partial match)

**Preprocessing**:
- Resize to fixed height (32px), maintain aspect ratio  
- CLAHE (clipLimit=3, tileGrid=(4,4))
- Otsu thresholding for binarization
- Morphological close to fill gaps
- Perspective deskew if extreme angle detected

**Post-processing**:
- Strip whitespace, convert to uppercase
- Character substitutions: O→0, I→1, S→5 (in numeric positions)
- Return "" if confidence < 0.3

---

## Violation Logic

```python
for each motorcycle M with associated riders R:
    num_riders = len(R)
    helmet_violations = sum(1 for r in R if r.helmet == "no_helmet")
    
    is_violation = (num_riders > 2) or (helmet_violations > 0)
    
    if is_violation:
        plate = detect_and_read_plate(M)
        violations.append({
            "num_riders": num_riders,
            "helmet_violations": helmet_violations,
            "license_plate": plate
        })
```

Only violating motorcycles are reported in output.

---

## File Structure

```
ROLL_NUMBER/
├── solution.py          ← TrafficViolationDetector class
├── models/
│   ├── yolov8n.pt       ← COCO detector (download)
│   ├── helmet_yolov8n.pt ← helmet detector (download from HF/github)
│   ├── lp_detector.pt   ← LP detector (download)
│   └── easyocr/         ← EasyOCR model cache (pre-downloaded)
├── requirements.txt
└── README.md

download_models.py       ← helper (run once before eval)
demo.py                  ← quick sanity check
report.md                ← analysis + failure cases
```

---

## Helmet Model Strategy

Since we can't train from scratch, we'll use:
1. **Primary**: Download a pre-trained helmet detector from a known public source:
   - `keremberke/yolov8n-helmet-detection` on HuggingFace (verified available)
   - ~6 MB ONNX or PT file
2. **Fallback**: OpenCV DNN head detection + HSV analysis

---

## Robustness Measures

| Challenge | Mitigation |
|---|---|
| Occlusions | Liberal IoA threshold (0.15), check horizontal overlap |
| Low lighting | CLAHE preprocessing; enhance contrast before OCR |
| Small plates | Bicubic upsampling to 4× before OCR |
| Multiple motorcycles | Hungarian assignment, each processed independently |
| No helmet visible | Count as no-helmet violation (conservative) |
| Runtime > 5s | Batch limit (max 5 motorcycles per image), resize input to 640 |

---

## Deliverables

1. `solution.py` — Full `TrafficViolationDetector` class
2. `requirements.txt` — Pinned dependencies
3. `README.md` — Setup + usage
4. `download_models.py` — One-click model download
5. `demo.py` — Test harness
6. `report.md` — System analysis, failure cases, design decisions
