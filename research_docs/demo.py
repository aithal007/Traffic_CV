"""
demo.py — Quick sanity-check + visualisation for TrafficViolationDetector

Usage:
    python demo.py <image_path>
    python demo.py <image_path> --show      # show annotated result
    python demo.py <image_path> --save out.jpg  # save annotated image
    python demo.py                          # generates a synthetic test image
"""

from __future__ import annotations

import os, sys, json, argparse, time
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "ROLL_NUMBER")
from solution import TrafficViolationDetector

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demo: Traffic Violation Detector")
parser.add_argument("image", nargs="?", default=None, help="Input image path")
parser.add_argument("--show",  action="store_true", help="Display result in a window")
parser.add_argument("--save",  default=None, help="Save annotated image to this path")
parser.add_argument("--model-dir", default="ROLL_NUMBER/models",
                    help="Path to models directory")
args = parser.parse_args()

# ── Generate synthetic image if no path given ─────────────────────────────────
if args.image is None:
    print("[INFO] No image provided — creating a synthetic test image")
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # grey background
    # Draw a fake motorcycle (dark rectangle)
    cv2.rectangle(img, (200, 250), (440, 400), (40, 40, 40), -1)
    cv2.ellipse(img,  (230, 400), (40, 20), 0, 0, 360, (20, 20, 20), -1)  # wheel
    cv2.ellipse(img,  (410, 400), (40, 20), 0, 0, 360, (20, 20, 20), -1)
    # Draw two fake persons (flesh-colour rectangles + circles for heads)
    for cx in [280, 350]:
        cv2.rectangle(img, (cx - 20, 180), (cx + 20, 260), (180, 140, 100), -1)  # body
        cv2.circle(img,    (cx, 165), 20, (180, 140, 100), -1)                   # head
    args.image = "test_synthetic.jpg"
    cv2.imwrite(args.image, img)
    print(f"[INFO] Saved to {args.image}")

# ── Run detector ──────────────────────────────────────────────────────────────
print(f"\nLoading model from '{args.model_dir}' ...")
t0 = time.time()
detector = TrafficViolationDetector(model_dir=args.model_dir)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.2f}s\n")

print(f"Running predict on '{args.image}' ...")
t1 = time.time()
result = detector.predict(args.image)
infer_time = time.time() - t1

print(f"Inference time : {infer_time:.2f}s")
print("\n── Output JSON ──────────────────────────────────────────────")
print(json.dumps(result, indent=2))
print("─────────────────────────────────────────────────────────────\n")

# ── Visualise if requested ────────────────────────────────────────────────────
if args.show or args.save:
    img = cv2.imread(args.image)
    if img is not None:
        h, w = img.shape[:2]
        for i, v in enumerate(result.get("violations", [])):
            text = (f"#{i+1} riders={v['num_riders']} "
                    f"helmets={v['helmet_violations']} "
                    f"plate={v['license_plate']}")
            cv2.putText(img, text, (10, 30 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if args.save:
            cv2.imwrite(args.save, img)
            print(f"[INFO] Saved annotated image to {args.save}")
        if args.show:
            cv2.imshow("Violations", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
