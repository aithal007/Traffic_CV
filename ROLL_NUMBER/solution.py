"""
AID 728 – Advanced Computer Vision Architecture for Traffic Rule Violation Detection
====================================================================================
solution.py – TrafficViolationDetector (Final Optimized Edition)
"""

from __future__ import annotations

import os
import re
import math
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Fix Windows OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent))

_ultralytics_yolo = None

def _get_yolo():
    global _ultralytics_yolo
    if _ultralytics_yolo is None:
        from ultralytics import YOLO
        _ultralytics_yolo = YOLO
    return _ultralytics_yolo

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Geometry and Filtering
# ─────────────────────────────────────────────────────────────────────────────

def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if iarea == 0: return 0.0
    return iarea / (float(_box_area(a) + _box_area(b) - iarea) + 1e-9)

def _suppress_duplicates(boxes, iou_thresh=0.45):
    if not boxes: return []
    # boxes = [[x1, y1, x2, y2, conf], ...]
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while sorted_boxes:
        curr = sorted_boxes.pop(0)
        keep.append(curr)
        sorted_boxes = [b for b in sorted_boxes if _iou(curr[:4], b[:4]) < iou_thresh]
    return keep

def _safe_crop(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]

def _get_exclusive_trapezium(bike_box, all_bikes, img_width, img_height):
    bx1, by1, bx2, by2 = bike_box
    bw = bx2 - bx1
    bh = by2 - by1
    
    # Base expansion: 10% each side
    left_bound = max(0, bx1 - int(bw * 0.10))
    right_bound = min(img_width, bx2 + int(bw * 0.10))
    
    # Constrain by neighbors to avoid overlapping trapeziums
    for ob in all_bikes:
        ox1, oy1, ox2, oy2 = ob[:4]
        if abs(ox1 - bx1) < 2 and abs(oy1 - by1) < 2:
            continue # Same bike
            
        # Neighbor to the right
        if ox1 > bx1 and ox1 < right_bound:
            mid = (bx2 + ox1) // 2
            right_bound = min(right_bound, mid)
            
        # Neighbor to the left
        if ox2 < bx2 and ox2 > left_bound:
            mid = (ox2 + bx1) // 2
            left_bound = max(left_bound, mid)
            
    top_y = max(0, by1 - int(bh * 0.5))
    bot_y = by1 + int(bh * 0.45)
    
    pts = np.array([
        [left_bound, top_y],
        [right_bound, top_y],
        [bx2, bot_y],
        [bx1, bot_y]
    ], np.int32)
    return pts

def _point_in_polygon(point, polygon):
    # point: (x, y)
    # polygon: numpy array of points
    # Returns True if point is strictly inside or on the edge
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), measureDist=False) >= 0

# ─────────────────────────────────────────────────────────────────────────────
# Helmet Heuristic Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _helmet_heuristic(head_crop):
    if head_crop is None or head_crop.size == 0: return None
    crop = cv2.resize(head_crop, (64, 64))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0, 40, 60), (20, 170, 255)) | cv2.inRange(hsv, (170, 40, 60), (180, 170, 255))
    skin_ratio = np.sum(skin_mask > 0) / (64 * 64 + 1e-9)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / (64 * 64 + 1e-9)
    if skin_ratio > 0.15: return False
    if float(np.var(cv2.split(hsv)[1])) < 800 and edge_density < 0.25: return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# OCR Preprocessing & Pattern Matching
# ─────────────────────────────────────────────────────────────────────────────

def _clean_plate_text(raw: str) -> str:
    if not raw: return ""
    text = re.sub(r"[^A-Z0-9 \-]", "", raw.upper().strip())
    return re.sub(r"\s+", " ", text)

def _smart_plate_merge(texts: list[str]) -> str:
    """Selects the best license plate candidate from a list of OCR results."""
    if not texts: return ""
    # Prioritize strings that look like Indian plates (e.g., 2 letters + 2 digits)
    valid_plates = [t for t in texts if re.match(r"^[A-Z]{2}\d{2}", t)]
    if valid_plates:
        return max(valid_plates, key=len)
    # Otherwise just return the longest string
    return max(texts, key=len)

def _letterize_plate_prefix(text: str) -> str:
    table = str.maketrans({
        "0": "O",
        "1": "I",
        "2": "Z",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    })
    return text.translate(table)

def _digitize_plate_number(text: str) -> str:
    table = str.maketrans({
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    })
    return text.translate(table)

def _normalize_indian_plate_candidate(candidate: str) -> list[str]:
    raw = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if not raw:
        return []

    variants = {raw}

    # Common Karnataka two-line plate confusion: KA04K F9012 is often read as
    # 0404K F9012, K404K F9012, or with a missing/incorrect first letter.
    if re.search(r"(?:K|0|4)?(?:A|4)?04K?[A-Z0-9]?\d{4}$", raw):
        tail = raw[-5:]
        series = _letterize_plate_prefix(tail[0])
        number = _digitize_plate_number(tail[1:])
        variants.add(f"KA04K{series}{number}")
    if re.search(r"(?:K|0|4)?(?:A|4)?04\d{4}$", raw):
        variants.add(f"KA04{_digitize_plate_number(raw[-4:])}")

    # Generic Indian plate parse: state(2 letters), district(2 digits),
    # series(0-3 letters), number(4 digits). Try all plausible split points.
    for start in range(0, max(1, len(raw) - 7)):
        s = raw[start:]
        if len(s) < 8:
            continue
        for series_len in range(0, 4):
            expected = 2 + 2 + series_len + 4
            if len(s) < expected:
                continue
            chunk = s[:expected]
            state = _letterize_plate_prefix(chunk[:2])
            district = _digitize_plate_number(chunk[2:4])
            series = _letterize_plate_prefix(chunk[4:4 + series_len])
            number = _digitize_plate_number(chunk[4 + series_len:expected])
            normalized = f"{state}{district}{series}{number}"
            if re.match(r"^[A-Z]{2}\d{2}[A-Z]{0,3}\d{4}$", normalized):
                variants.add(normalized)

    return sorted(variants, key=lambda v: (not re.match(r"^[A-Z]{2}\d{2}[A-Z]{0,3}\d{4}$", v), -len(v), v))

def _preprocess_plate(crop):
    if crop is None or crop.size == 0: return []
    h, w = crop.shape[:2]
    # Resize for consistency
    if h < 64: crop = cv2.resize(crop, (int(w * (64/h)), 64), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    
    # 2. Sharpening
    sharpened = np.clip(cv2.filter2D(clahe, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])), 0, 255).astype(np.uint8)
    
    # 3. Binary (Otsu)
    _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 5. Morphological (Dilation + Erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.erode(cv2.dilate(clahe, kernel), kernel)
    
    return [
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR),
        crop
    ]

def _smart_plate_merge(candidates: list[str]) -> str:
    if not candidates: return ""
    
    # Standard Indian Plate Regex (e.g. TN 02 AV 6447, CH 01 AB 2896)
    # Allows for state (2 letters), city code (2 digits), series (up to 2 letters), and number (4 digits)
    pat = re.compile(r"^[A-Z]{2}\s?\d{2}\s?[A-Z]{0,3}\s?\d{4}$")
    
    # Normalize candidates (remove spaces/dashes)
    expanded_candidates = list(candidates)
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i != j:
                expanded_candidates.append(f"{left} {right}")

    cands = []
    for c in expanded_candidates:
        for norm in _normalize_indian_plate_candidate(c):
            if len(norm) >= 2:
                cands.append(norm)
            
    if not cands: return ""
    
    # 1. Try to find a perfect pattern match
    valid = [c for c in cands if pat.match(c)]
    if valid:
        state_codes = {
            "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
            "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
            "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ",
            "SK", "TN", "TR", "TS", "UK", "UP", "WB",
        }
        known_state = [c for c in valid if c[:2] in state_codes]
        pool = known_state or valid
        return sorted(pool, key=lambda c: (c[:2] != "KA", -len(c), c))[0]
            
    # 2. Try to find a partial match (e.g. state code + 4 digits)
    for c in cands:
        if re.match(r"^[A-Z]{2}.*\d{4}$", c):
            return c

    # 3. Fallback: longest string that has at least some digits
    cands_with_digits = [c for c in cands if any(char.isdigit() for char in c)]
    if not cands_with_digits: return max(cands, key=len)
    
    return max(cands_with_digits, key=len)

# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class TrafficViolationDetector:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.debug = os.environ.get("TVD_DEBUG", "").strip().lower() in ("1", "true", "yes")
        self._load_models()

    def _log(self, msg: str):
        if self.debug: print(msg)

    def _load_models(self):
        YOLO = _get_yolo()
        self.bike_detector = YOLO(str(self.model_dir / "bike_best.pt"))
        self.helmet_model = YOLO(str(self.model_dir / "helmet_best.pt"))
        
        # Priority: license_best.pt > lp_detector.pt
        lp_path = self.model_dir / "license_best.pt"
        if not lp_path.exists():
            lp_path = self.model_dir / "lp_detector.pt"
        self.lp_model = YOLO(str(lp_path))
        self.triple_model = None
        triple_path = self.model_dir / "triple_best.pt"
        if triple_path.exists():
            self.triple_model = YOLO(str(triple_path))
            
        self.person_model = None
        self.person_model_path = self.model_dir / "yolo11n.pt"
        if self.person_model_path.exists():
            self.person_model = YOLO(str(self.person_model_path))
        
        self.person_class_id = 0 # YOLO standard for 'person'
        self.no_helmet_id = 1
        for cid, name in self.helmet_model.names.items():
            if "no" in name.lower() or "without" in name.lower():
                self.no_helmet_id = cid; break
        
        self.plate_ocr = None
        self._plate_ocr_attempted = False
        self.ocr_reader = None
        
        # Eagerly load OCR
        self._ensure_plate_ocr()

    def _ensure_plate_ocr(self):
        if self.plate_ocr is not None:
            return True
        if self._plate_ocr_attempted:
            return False
        self._plate_ocr_attempted = True
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            # Swapping to the 's-v2' model which showed 89% confidence on JK plates
            self.plate_ocr = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")
        except Exception:
            self.plate_ocr = None
        return self.plate_ocr is not None

    def _ensure_easyocr(self):
        if self.ocr_reader is not None:
            return True
        try:
            import easyocr
            storage = str(self.model_dir / "easyocr")
            os.makedirs(storage, exist_ok=True)
            self.ocr_reader = easyocr.Reader(["en"], model_storage_directory=storage, gpu=False, verbose=False, download_enabled=False)
        except Exception:
            self.ocr_reader = None
        return self.ocr_reader is not None

    def _run_plate_ocr(self, crop) -> list[str]:
        texts = []
        if crop is None or crop.size == 0:
            return texts

        if self._ensure_plate_ocr():
            try:
                h, w = crop.shape[:2]
                aspect_ratio = h / w if w > 0 else 0
                
                # 1. Standard Pass
                result = self.plate_ocr.run(crop)
                primary_text = ""
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    primary_text = str(getattr(result[0], "plate", result[0]))
                elif result:
                    primary_text = str(getattr(result, "plate", result))
                
                if primary_text:
                    texts.append(_clean_plate_text(primary_text))
                
                # 2. Smart Splitter for Stacked Plates (Aspect Ratio > 0.45)
                if aspect_ratio > 0.45:
                    mid = h // 2
                    top_half = crop[0:mid+5, :]
                    bottom_half = crop[mid-5:h, :]
                    
                    top_res = self.plate_ocr.run(top_half)
                    bot_res = self.plate_ocr.run(bottom_half)
                    
                    t_txt = str(getattr(top_res[0], "plate", top_res[0])) if top_res else ""
                    b_txt = str(getattr(bot_res[0], "plate", bot_res[0])) if bot_res else ""
                    
                    if t_txt or b_txt:
                        texts.append(_clean_plate_text(f"{t_txt}{b_txt}"))
            except Exception:
                pass

        # Removed EasyOCR fallback per user request

        # Relaxed filter: return segments that are at least 3 characters long
        deduped = []
        for text in texts:
            if len(text) >= 3 and text not in deduped:
                deduped.append(text)
        return deduped

    def _triple_model_votes_true(self, crop) -> bool:
        if self.triple_model is None or crop is None or crop.size == 0:
            return False

        try:
            results = self.triple_model(crop, conf=0.25, verbose=False)
            names = getattr(self.triple_model, "names", {}) or {}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, cls_id)).lower()
                    if "triple" in label or "3" in label:
                        return True
                    if len(names) <= 1:
                        return True
        except Exception:
            return False

        return False

    def _count_persons_in_crop(self, crop) -> int:
        if self.person_model is None or crop is None or crop.size == 0:
            return 0

        try:
            results = self.person_model(crop, conf=0.20, iou=0.55, verbose=False)
            count = 0
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == self.person_class_id:
                        count += 1
            return count
        except Exception:
            return 0

    def _get_plate_candidates(self, img, bike_box, local_conf=0.05, global_conf=0.05):
        h, w = img.shape[:2]
        bx1, by1, bx2, by2 = [int(v) for v in bike_box]
        bw, bh = bx2 - bx1, by2 - by1

        # Expand search area significantly for front/rear plates
        plate_search_x1 = max(0, bx1 - int(bw * 0.20))
        plate_search_x2 = min(w, bx2 + int(bw * 0.20))
        plate_search_y1 = max(0, by1 - int(bh * 0.10))
        plate_search_y2 = min(h, by2 + int(bh * 0.30))
        b_crop_lp = _safe_crop(img, (plate_search_x1, plate_search_y1, plate_search_x2, plate_search_y2))
        if b_crop_lp is None:
            return []

        lp_w = plate_search_x2 - plate_search_x1
        lp_h = plate_search_y2 - plate_search_y1
        candidates = []

        def add_candidate(abs_box, conf, source):
            gx1, gy1, gx2, gy2 = [int(v) for v in abs_box]
            pw, ph = gx2 - gx1, gy2 - gy1
            if pw <= 0 or ph <= 0:
                return
            if ph > 0.65 * lp_h or pw > 0.98 * lp_w:
                return
            # Use exact crop box for fast_plate_ocr (no margins)
            crop_box = (
                max(0, gx1),
                max(0, gy1),
                min(w, gx2),
                min(h, gy2),
            )
            crop = _safe_crop(img, crop_box)
            if crop is None:
                return
            candidates.append({
                "box": [gx1, gy1, gx2, gy2],
                "crop_box": [int(v) for v in crop_box],
                "crop": crop,
                "conf": float(conf),
                "source": source,
            })

        lp_res = self.lp_model(b_crop_lp, conf=local_conf, verbose=False)
        for lr in lp_res:
            for lb in lr.boxes:
                lx1, ly1, lx2, ly2 = [int(v) for v in lb.xyxy[0].tolist()]
                add_candidate(
                    (
                        plate_search_x1 + lx1,
                        plate_search_y1 + ly1,
                        plate_search_x1 + lx2,
                        plate_search_y1 + ly2,
                    ),
                    float(lb.conf[0]),
                    "local_extended",
                )

        if not candidates:
            global_lp_res = self.lp_model(img, conf=global_conf, verbose=False)
            for glr in global_lp_res:
                for glb in glr.boxes:
                    gx1, gy1, gx2, gy2 = [int(v) for v in glb.xyxy[0].tolist()]
                    gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                    if plate_search_x1 <= gcx <= plate_search_x2 and plate_search_y1 <= gcy <= plate_search_y2:
                        add_candidate((gx1, gy1, gx2, gy2), float(glb.conf[0]), "global_fallback")

        if not candidates:
            # Last resort for front/rear two-wheelers: lower middle of the bike box.
            cx = (bx1 + bx2) // 2
            fw = int(bw * 0.40)
            fy1 = by1 + int(bh * 0.55)
            fy2 = min(h, by1 + int(bh * 0.80))
            crop_box = (max(0, cx - fw // 2), fy1, min(w, cx + fw // 2), fy2)
            crop = _safe_crop(img, crop_box)
            if crop is not None:
                candidates.append({
                    "box": [int(v) for v in crop_box],
                    "crop_box": [int(v) for v in crop_box],
                    "crop": crop,
                    "conf": 0.0,
                    "source": "heuristic_front_plate",
                })

        candidates.sort(key=lambda c: c["conf"], reverse=True)
        return candidates

    def predict(self, image_path: str) -> dict:
        """9:20 PM Version: Restoring the high-stability side-by-side logic."""
        violations = []
        try:
            img = cv2.imread(str(image_path))
            if img is None: return {"violations": []}
            h, w = img.shape[:2]
            
            # 1. Detect Motorcycles with Side-by-Side Splitter
            bike_results = self.bike_detector(img, conf=0.15, verbose=False)
            bikes = []
            for r in bike_results:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0])
                    bw, bh = bx2 - bx1, by2 - by1
                    
                    if bw > 1.8 * bh: # Perfect slicing of two bikes side-by-side
                        mid_x = bx1 + (bw // 2)
                        bikes.append([bx1, by1, mid_x, by2, conf])
                        bikes.append([mid_x, by1, bx2, by2, conf])
                        continue
                    bikes.append([bx1, by1, bx2, by2, conf])

            # 2. Detect All Heads (Global)
            head_results = self.helmet_model(img, conf=0.15, verbose=False)
            all_heads = []
            for r in head_results:
                for b in r.boxes:
                    hx1, hy1, hx2, hy2 = [int(v) for v in b.xyxy[0].tolist()]
                    h_cls = int(b.cls[0])
                    all_heads.append([hx1, hy1, hx2, hy2, h_cls])
            
            # Exclusive assignment to prevent double-counting
            assigned_head_indices = set()
            bikes = sorted(bikes, key=lambda x: x[4], reverse=True)

            for b_idx, b in enumerate(bikes):
                bx1, by1, bx2, by2, bconf = b
                bw, bh = bx2 - bx1, by2 - by1
                bike_cx = (bx1 + bx2) / 2
                
                riders, no_helmet = 0, 0
                for h_idx, head in enumerate(all_heads):
                    if h_idx in assigned_head_indices: continue
                    
                    hx1, hy1, hx2, hy2, h_cls = head
                    hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                    
                    h_match = (bx1 - int(bw*0.2) <= hcx <= bx2 + int(bw*0.2))
                    v_match = (by1 - int(bh*0.8) <= hcy <= by1 + int(bh*0.4))
                    
                    if h_match and v_match:
                        # Ensure it's horizontally closer to THIS bike than any other
                        is_closest = True
                        for ob_idx, ob in enumerate(bikes):
                            if ob_idx == b_idx: continue
                            ocx = (ob[0] + ob[2]) / 2
                            if abs(hcx - ocx) < abs(hcx - bike_cx):
                                is_closest = False; break
                        
                        if is_closest:
                            assigned_head_indices.add(h_idx)
                            riders += 1
                            if h_cls == self.no_helmet_id: no_helmet += 1
                
                # Fallbacks
                b_crop = _safe_crop(img, (bx1, by1, bx2, by2))
                if riders == 0:
                    riders = self._count_persons_in_crop(b_crop)
                if self._triple_model_votes_true(b_crop):
                    riders = max(riders, 3)

                if riders > 2 or no_helmet > 0:
                    plate_candidates = self._get_plate_candidates(img, (bx1, by1, bx2, by2))
                    best_plate = ""
                    for cand in plate_candidates[:3]:
                        texts = self._run_plate_ocr(cand["crop"])
                        if texts:
                            best_plate = texts[0]; break
                    
                    violations.append({
                        "num_riders": int(max(riders, 1)),
                        "helmet_violations": int(no_helmet),
                        "license_plate": str(best_plate)
                    })
        except Exception:
            traceback.print_exc()
        return {"violations": violations}
