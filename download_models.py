"""
download_models.py
──────────────────
Downloads all required model weights into ROLL_NUMBER/models/.

Usage (run ONCE before evaluation — requires internet):
    python download_models.py

What it downloads:
  1. yolo11n.pt          — YOLO11-nano COCO detector (~6 MB)
  2. helmet_yolov8n.pt   — YOLOv8-nano helmet classifier (~6 MB)
  3. lp_detector.pt      — YOLOv8-nano license-plate detector (~6 MB)
  4. EasyOCR English models (~90 MB, cached in models/easyocr/)

Total: ~108 MB  ✓ (well under 250 MB limit)
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path

MODELS_DIR = Path("ROLL_NUMBER/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EASYOCR_DIR = MODELS_DIR / "easyocr"
EASYOCR_DIR.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, description: str):
    if dest.exists():
        print(f"  [SKIP] {description} already exists ({dest.stat().st_size // 1024} KB)")
        return True
    print(f"  [DOWN] {description} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  [OK]   {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


print("=" * 60)
print("Traffic Violation Detector — Model Downloader")
print("=" * 60)

# ── 1. YOLO11n COCO detector ──────────────────────────────────────────────
print("\n[1/4] YOLO11n COCO detector")
coco_ok = download(
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    MODELS_DIR / "yolo11n.pt",
    "yolo11n.pt (COCO)"
)

# ── 2. Helmet detector ────────────────────────────────────────────────────
print("\n[2/4] Helmet detector (YOLOv8n fine-tuned)")
# Primary source: keremberke's HuggingFace model
helmet_sources = [
    (
        "https://huggingface.co/keremberke/yolov8n-helmet-detection/resolve/main/best.pt",
        "HuggingFace keremberke/yolov8n-helmet-detection"
    ),
    (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "Fallback: using COCO model (helmet classification will use heuristic)"
    ),
]
helmet_ok = False
helmet_dest = MODELS_DIR / "helmet_yolov8n.pt"
for url, desc in helmet_sources:
    if download(url, helmet_dest, desc):
        helmet_ok = True
        break

# ── 3. License-plate detector ─────────────────────────────────────────────
print("\n[3/4] License-plate detector (YOLOv8n fine-tuned)")
lp_sources = [
    (
        "https://huggingface.co/keremberke/yolov8n-license-plate-detection/resolve/main/best.pt",
        "HuggingFace keremberke/yolov8n-license-plate-detection"
    ),
    (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "Fallback: using COCO (LP detection will fall back to bottom-region heuristic)"
    ),
]
lp_ok = False
lp_dest = MODELS_DIR / "lp_detector.pt"
for url, desc in lp_sources:
    if download(url, lp_dest, desc):
        lp_ok = True
        break

# ── 4. EasyOCR English models ─────────────────────────────────────────────
print("\n[4/4] EasyOCR English language models")
try:
    import easyocr
    print("  Initialising EasyOCR (this downloads ~90 MB on first run)...")
    reader = easyocr.Reader(
        ["en"],
        model_storage_directory=str(EASYOCR_DIR),
        gpu=False,
        verbose=True,
        download_enabled=True,
    )
    print("  [OK] EasyOCR ready")
    del reader
except ImportError:
    print("  [FAIL] easyocr not installed. Run: pip install easyocr")
except Exception as e:
    print(f"  [WARN] EasyOCR init: {e}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Download summary:")
total_mb = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()) / 1e6
print(f"  Total model size : {total_mb:.1f} MB  (limit: 250 MB)")
ok = lambda p: "OK" if p else "MISSING"
print(f"  COCO detector    : {ok((MODELS_DIR/'yolo11n.pt').exists())}")
print(f"  Helmet detector  : {ok((MODELS_DIR/'helmet_yolov8n.pt').exists())}")
print(f"  LP detector      : {ok((MODELS_DIR/'lp_detector.pt').exists())}")
print(f"  EasyOCR models   : {ok(any(EASYOCR_DIR.iterdir()))}")
print("=" * 60)
print("Run 'python demo.py <image>' to test the system.")
