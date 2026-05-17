"""
Visualization for the 4-stage traffic violation pipeline.
Uses the SAME expanded-crop logic as solution.py for accurate results.

Color scheme:
  Blue   = Bike bounding box
  Green  = Rider WITH helmet
  Red    = Rider WITHOUT helmet  (violation)
  Cyan   = License plate box
"""
import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "ROLL_NUMBER"))

from ultralytics import YOLO
from solution import _helmet_heuristic, _safe_crop

models_dir   = "ROLL_NUMBER/models"
bike_model   = YOLO(os.path.join(models_dir, "bike_best.pt"))
helmet_model = YOLO(os.path.join(models_dir, "helmet_best.pt"))
lp_model     = YOLO(os.path.join(models_dir, "lp_detector.pt"))

img = cv2.imread("image.png")
vis = img.copy()
img_h, img_w = img.shape[:2]

NO_HELMET_ID = 1  # class 1 = no_helmet in helmet_best.pt

print(f"Helmet model classes: {helmet_model.names}")

# ── Stage 1: detect bikes ─────────────────────────────────────────────────────
bike_results = bike_model(img, conf=0.55, verbose=False)
raw_bikes = []
for r in bike_results:
    for b in r.boxes:
        bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
        raw_bikes.append([bx1, by1, bx2, by2, float(b.conf[0])])

# Use the suppression logic from solution.py
from solution import _suppress_duplicates
bikes = _suppress_duplicates(raw_bikes, iou_thresh=0.60)

print(f"\nDetected {len(bikes)} unique bikes\n")

# 1. PRE-DETECT ALL HEADS GLOBALLY
all_heads_global = []
h_full_res = helmet_model(img, conf=0.12, verbose=False)
raw_global_heads = []
for hr in h_full_res:
    for hb in hr.boxes:
        hx1, hy1, hx2, hy2 = [int(v) for v in hb.xyxy[0].tolist()]
        raw_global_heads.append([hx1, hy1, hx2, hy2, float(hb.conf[0]), int(hb.cls[0])])
all_heads_global = _suppress_duplicates(raw_global_heads, iou_thresh=0.20)

assigned_head_indices = set()
bikes = sorted(bikes, key=lambda x: x[4], reverse=True)

for b in bikes:
    bx1, by1, bx2, by2, bconf = b
    bh, bw = by2 - by1, bx2 - bx1

    b_crop = _safe_crop(img, (bx1, by1, bx2, by2))
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 100, 0), 3)

    # ── Stage 2: GREEDY HEAD ASSOCIATION ──────────────────────────────────
    num_riders = 0
    no_helmet  = 0
    
    zone_y1 = max(0, by1 - int(bh * 1.0))
    zone_y2 = by1 + int(bh * 0.2)

    for h_idx, h in enumerate(all_heads_global):
        if h_idx in assigned_head_indices: continue
        
        hx1, hy1, hx2, hy2, h_conf, h_cls = h
        hh = hy2 - hy1
        
        if hh < (0.06 * bh) or hh > (0.50 * bh): continue
        
        h_cx = (hx1 + hx2) // 2
        h_cy = (hy1 + hy2) // 2
        
        if bx1 <= h_cx <= bx2 and zone_y1 <= h_cy <= zone_y2:
            assigned_head_indices.add(h_idx)
            num_riders += 1
            
            is_nh = False
            head_box_crop = _safe_crop(img, (hx1, hy1, hx2, hy2))
            if h_conf > 0.60:
                is_nh = (h_cls == NO_HELMET_ID)
            else:
                heur = _helmet_heuristic(head_box_crop)
                if heur is False: is_nh = True
                elif heur is None: is_nh = (NO_HELMET_ID == h_cls)
            
            if is_nh: no_helmet += 1
            
            color = (0, 0, 220) if is_nh else (0, 200, 0)
            label = f"{'NH' if is_nh else 'H'} {h_conf:.2f}"
            cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(vis, label, (hx1, max(hy1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    # Bike label with violation summary
    is_violation = no_helmet > 0 or num_riders > 2
    status       = "VIOLATION" if is_violation else "OK"
    bike_color   = (0, 0, 220) if is_violation else (0, 200, 0)
    bike_label   = f"BIKE | {num_riders}R {no_helmet}NH | {status}"
    cv2.putText(vis, bike_label, (bx1, max(by1 - 14, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bike_color, 2)

    print(f"  Bike@({bx1},{by1}): riders={num_riders}  no_helmet={no_helmet}  {'⚠ VIOLATION' if is_violation else 'OK'}")

    # ── Stage 3: plate detection on tight crop ────────────────────────────
    if b_crop is not None:
        lp_res = lp_model(b_crop, conf=0.10, verbose=False)
        for lr in lp_res:
            for lb in lr.boxes:
                lx1, ly1, lx2, ly2 = [int(v) for v in lb.xyxy[0].tolist()]
                lh, lw = ly2 - ly1, lx2 - lx1
                if lh > 0.40 * bh or lw > 0.80 * bw:
                    continue  # reject oversized
                px1, py1 = bx1 + lx1, by1 + ly1
                px2, py2 = bx1 + lx2, by1 + ly2
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 220, 220), 2)
                cv2.putText(vis, "PLATE", (px1, max(py1 - 5, 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 2)
                print(f"    Plate @ ({px1},{py1})-({px2},{py2})")

out = "final_output_visual.jpg"
cv2.imwrite(out, vis)
print(f"\nSaved → {out}")
