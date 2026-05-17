from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLL_DIR = PROJECT_ROOT / "ROLL_NUMBER"
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROLL_DIR))

from ROLL_NUMBER.solution import (  # noqa: E402
    TrafficViolationDetector,
    _get_exclusive_trapezium,
    _clean_plate_text,
    _helmet_heuristic,
    _point_in_polygon,
    _safe_crop,
    _smart_plate_merge,
    _suppress_duplicates,
)


def draw_label(img, box, label, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_y = max(0, y1 - th - base - 5)
    cv2.rectangle(img, (x1, label_y), (x1 + tw + 6, label_y + th + base + 5), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, label_y + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def parse_bikes(detector, img):
    bike_results = detector.bike_detector(img, conf=0.30, iou=0.70, verbose=False)
    bikes = []
    for r in bike_results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1
            if bw > 1.8 * bh:
                mid_x = x1 + bw // 2
                bikes.append([x1, y1, mid_x, y2, conf, "split"])
                bikes.append([mid_x, y1, x2, y2, conf, "split"])
            else:
                bikes.append([x1, y1, x2, y2, conf, "raw"])
    return sorted(bikes, key=lambda item: item[4], reverse=True)


def parse_heads(detector, img):
    attempts = []
    raw_heads = []
    for conf in (0.30, 0.10):
        current = []
        helmet_results = detector.helmet_model(img, conf=conf, verbose=False)
        for r in helmet_results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                current.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])
        attempts.append({"conf": conf, "count": len(current), "boxes": current})
        raw_heads.extend(current)

    return {
        "used_conf": "merged_0.30_and_0.10",
        "attempts": attempts,
        "heads": _suppress_duplicates(raw_heads, iou_thresh=0.20),
        "names": detector.helmet_model.names,
    }


def run_ocr_variants(detector, crop):
    readings = []
    if crop is None or crop.size == 0:
        return readings

    if detector.plate_ocr:
        h, w = crop.shape[:2]
        candidates = [(1.0, crop)]
        if w < 80:
            candidates.append((2.0, cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)))
        for scale, candidate in candidates:
            texts = detector._run_plate_ocr(candidate)
            readings.append({"engine": "fast_plate_ocr", "scale": scale, "texts": texts})
        return readings

    if detector.ocr_reader:
        for scale in (1.0, 2.0, 3.0):
            h, w = crop.shape[:2]
            candidate = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            helper_texts = detector._run_plate_ocr(candidate)
            raw_items = []
            try:
                raw = detector.ocr_reader.readtext(candidate, detail=1)
                for item in sorted(raw, key=lambda x: x[0][0][1]):
                    raw_items.append(
                        {
                            "text": _clean_plate_text(item[1]),
                            "confidence": float(item[2]),
                            "bbox": [[int(p[0]), int(p[1])] for p in item[0]],
                        }
                    )
            except Exception as exc:
                raw_items.append({"error": str(exc)})
            readings.append(
                {
                    "engine": "easyocr",
                    "scale": scale,
                    "texts": helper_texts,
                    "raw": raw_items,
                }
            )

    return readings


def inspect_plate_stage(detector, img, bike, out_dir, bike_idx):
    h, w = img.shape[:2]
    bx1, by1, bx2, by2, _, _ = bike
    bw, bh = bx2 - bx1, by2 - by1

    plate_search_x1 = max(0, bx1 - int(bw * 0.03))
    plate_search_x2 = min(w, bx2 + int(bw * 0.03))
    plate_search_y2 = min(h, by2 + int(bh * 0.15))
    lp_crop = _safe_crop(img, (plate_search_x1, by1, plate_search_x2, plate_search_y2))
    lp_w = plate_search_x2 - plate_search_x1
    lp_h = plate_search_y2 - by1

    candidate_infos = []
    if lp_crop is not None:
        search_path = out_dir / f"bike_{bike_idx:02d}_lp_search.png"
        cv2.imwrite(str(search_path), lp_crop)
        lp_results = detector.lp_model(lp_crop, conf=0.10, verbose=False)
        for result in lp_results:
            for box in result.boxes:
                lx1, ly1, lx2, ly2 = [int(v) for v in box.xyxy[0].tolist()]
                pw, ph = lx2 - lx1, ly2 - ly1
                mx, my = int(pw * 0.25), int(ph * 0.25)
                ex1, ey1 = max(0, lx1 - mx), max(0, ly1 - my)
                ex2, ey2 = min(lp_w, lx2 + mx), min(lp_h, ly2 + my)
                rejected = ph > 0.40 * lp_h or pw > 0.80 * lp_w
                plate_crop = None if rejected else _safe_crop(lp_crop, (ex1, ey1, ex2, ey2))
                crop_path = ""
                readings = []
                if plate_crop is not None:
                    crop_path = str(out_dir / f"bike_{bike_idx:02d}_plate_{len(candidate_infos) + 1:02d}.png")
                    cv2.imwrite(crop_path, plate_crop)
                    readings = run_ocr_variants(detector, plate_crop)
                candidate_infos.append(
                    {
                        "lp_box_in_extended_crop": [lx1, ly1, lx2, ly2],
                        "lp_box_in_image": [
                            plate_search_x1 + lx1,
                            by1 + ly1,
                            plate_search_x1 + lx2,
                            by1 + ly2,
                        ],
                        "confidence": float(box.conf[0]),
                        "expanded_crop_box": [ex1, ey1, ex2, ey2],
                        "rejected": rejected,
                        "crop_path": crop_path,
                        "ocr": readings,
                    }
                )

    if not any(not candidate["rejected"] for candidate in candidate_infos):
        global_lp_results = detector.lp_model(img, conf=0.15, verbose=False)
        for result in global_lp_results:
            for box in result.boxes:
                gx1, gy1, gx2, gy2 = [int(v) for v in box.xyxy[0].tolist()]
                gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                if not (plate_search_x1 <= gcx <= plate_search_x2 and by1 <= gcy <= plate_search_y2):
                    continue
                pw, ph = gx2 - gx1, gy2 - gy1
                rejected = ph > 0.40 * lp_h or pw > 0.80 * lp_w
                plate_crop = None
                crop_path = ""
                readings = []
                if not rejected:
                    mx, my = int(pw * 0.25), int(ph * 0.25)
                    plate_crop = _safe_crop(
                        img,
                        (
                            max(0, gx1 - mx),
                            max(0, gy1 - my),
                            min(w, gx2 + mx),
                            min(h, gy2 + my),
                        ),
                    )
                    if plate_crop is not None:
                        crop_path = str(out_dir / f"bike_{bike_idx:02d}_plate_global_{len(candidate_infos) + 1:02d}.png")
                        cv2.imwrite(crop_path, plate_crop)
                        readings = run_ocr_variants(detector, plate_crop)
                candidate_infos.append(
                    {
                        "source": "global_lp_fallback",
                        "lp_box_in_extended_crop": [
                            gx1 - plate_search_x1,
                            gy1 - by1,
                            gx2 - plate_search_x1,
                            gy2 - by1,
                        ],
                        "lp_box_in_image": [gx1, gy1, gx2, gy2],
                        "confidence": float(box.conf[0]),
                        "expanded_crop_box": [],
                        "rejected": rejected,
                        "crop_path": crop_path,
                        "ocr": readings,
                    }
                )

    fallback_info = None
    if not any(not candidate["rejected"] for candidate in candidate_infos) and lp_crop is not None:
        fallback = _safe_crop(lp_crop, (0, int(lp_h * 0.60), lp_w, lp_h))
        if fallback is not None:
            fallback_path = out_dir / f"bike_{bike_idx:02d}_plate_fallback.png"
            cv2.imwrite(str(fallback_path), fallback)
            fallback_info = {
                "crop_path": str(fallback_path),
                "ocr": run_ocr_variants(detector, fallback),
            }

    all_texts = []
    for candidate in candidate_infos:
        if candidate["rejected"]:
            continue
        for reading in candidate["ocr"]:
            all_texts.extend(reading.get("texts", []))
    if fallback_info:
        for reading in fallback_info["ocr"]:
            all_texts.extend(reading.get("texts", []))

    return {
        "extended_search_box": [plate_search_x1, by1, plate_search_x2, plate_search_y2],
        "extended_crop_shape": None if lp_crop is None else list(lp_crop.shape),
        "candidates": candidate_infos,
        "fallback": fallback_info,
        "all_ocr_texts": all_texts,
        "merged_plate": _smart_plate_merge(all_texts),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the full bike/helmet/LP/OCR pipeline with debug artifacts.")
    parser.add_argument("image", nargs="?", default="image.png", help="Image to test, e.g. image.png or imag.png")
    parser.add_argument("--model-dir", default=str(ROLL_DIR / "models"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "pipeline_debug"))
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detector = TrafficViolationDetector(args.model_dir)
    h, w = img.shape[:2]
    vis = img.copy()

    bikes = parse_bikes(detector, img)
    head_data = parse_heads(detector, img)
    assigned_head_indices = set()
    bike_reports = []

    for bike_idx, bike in enumerate(bikes, 1):
        bx1, by1, bx2, by2, bike_conf, source = bike
        bw, bh = bx2 - bx1, by2 - by1
        trapezium = _get_exclusive_trapezium((bx1, by1, bx2, by2), bikes, w, h)
        cv2.polylines(vis, [trapezium], True, (255, 180, 0), 2)
        draw_label(vis, (bx1, by1, bx2, by2), f"bike {bike_idx} {bike_conf:.2f}", (0, 200, 255))

        assigned_heads = []
        no_helmet_count = 0
        for head_idx, head in enumerate(head_data["heads"]):
            if head_idx in assigned_head_indices:
                continue
            hx1, hy1, hx2, hy2, head_conf, head_cls = head
            center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
            if _point_in_polygon(center, trapezium):
                assigned_head_indices.add(head_idx)
                name = str(head_data["names"].get(head_cls, head_cls))
                head_crop = _safe_crop(img, (hx1, hy1, hx2, hy2))
                heuristic = _helmet_heuristic(head_crop)
                is_no_helmet = head_cls == detector.no_helmet_id if head_conf > 0.60 else heuristic is False
                no_helmet_count += int(is_no_helmet)
                assigned_heads.append(
                    {
                        "box": [hx1, hy1, hx2, hy2],
                        "confidence": head_conf,
                        "class_id": head_cls,
                        "class_name": name,
                        "center": list(center),
                        "heuristic_has_helmet": heuristic,
                        "is_no_helmet": bool(is_no_helmet),
                    }
                )

        b_crop = _safe_crop(img, (bx1, by1, bx2, by2))
        person_count = detector._count_persons_in_crop(b_crop)
        triple_vote = detector._triple_model_votes_true(b_crop)
        rider_count = len(assigned_heads)
        if rider_count == 0:
            rider_count = person_count
        if triple_vote:
            rider_count = max(rider_count, 3)

        plate_report = inspect_plate_stage(detector, img, bike, out_dir, bike_idx)
        for candidate in plate_report["candidates"]:
            x1, y1, x2, y2 = candidate["lp_box_in_image"]
            draw_label(vis, (x1, y1, x2, y2), f"lp {candidate['confidence']:.2f}", (255, 0, 255))

        bike_reports.append(
            {
                "bike_index": bike_idx,
                "bike_box": [bx1, by1, bx2, by2],
                "bike_confidence": bike_conf,
                "box_source": source,
                "trapezium": trapezium.tolist(),
                "assigned_heads": assigned_heads,
                "head_based_rider_count": len(assigned_heads),
                "person_count_floor": person_count,
                "triple_model_vote": triple_vote,
                "final_rider_count_before_violation_cap": rider_count,
                "helmet_violations": no_helmet_count,
                "plate": plate_report,
                "would_emit_violation": rider_count > 2 or no_helmet_count > 0,
            }
        )

    for head_idx, head in enumerate(head_data["heads"]):
        hx1, hy1, hx2, hy2, conf, cls_id = head
        name = str(head_data["names"].get(cls_id, cls_id))
        color = (0, 0, 255) if cls_id == detector.no_helmet_id else (0, 255, 0)
        draw_label(vis, (hx1, hy1, hx2, hy2), f"{name} {conf:.2f}", color)

    annotated_path = out_dir / f"{image_path.stem}_annotated_pipeline.png"
    cv2.imwrite(str(annotated_path), vis)

    prediction = detector.predict(str(image_path))
    report = {
        "image": str(image_path),
        "image_shape": [h, w, int(img.shape[2])],
        "ocr_engine": "fast_plate_ocr" if detector.plate_ocr else "easyocr" if detector.ocr_reader else "none",
        "helmet_detection": head_data,
        "bikes": bike_reports,
        "predict_output": prediction,
        "annotated_image": str(annotated_path),
    }

    report_path = out_dir / f"{image_path.stem}_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved annotated image: {annotated_path}")


if __name__ == "__main__":
    main()
