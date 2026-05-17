import sys
import cv2
import json
import numpy as np
from pathlib import Path

# Add ROLL_NUMBER to path so we can import the class
sys.path.append(str(Path("ROLL_NUMBER").absolute()))
from solution import TrafficViolationDetector, _safe_crop, _smart_plate_merge

def test_lp_and_ocr(img_path, out_path):
    print(f"--- LP & OCR TEST ON {img_path} ---")
    
    detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return

    # 1. Detect Motorcycles
    bike_res = detector.bike_detector(img, conf=0.15, verbose=False)
    
    for r in bike_res:
        for b in r.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            bw, bh = bx2 - bx1, by2 - by1
            
            # Draw Bike
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            
            for cand in detector._get_plate_candidates(img, (bx1, by1, bx2, by2)):
                gx1, gy1, gx2, gy2 = cand["box"]
                cx1, cy1, cx2, cy2 = cand["crop_box"]
                ocr_texts = detector._run_plate_ocr(cand["crop"])
                ocr_result = _smart_plate_merge(ocr_texts) if ocr_texts else "OCR_FAIL"

                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1)
                cv2.putText(img, f"LP: {ocr_result}", (gx1, max(20, gy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Found Plate: {ocr_result} raw={ocr_texts} source={cand['source']} at {[gx1, gy1, gx2, gy2]}")

    cv2.imwrite(out_path, img)
    print(f"Saved result to {out_path}")

if __name__ == "__main__":
    target = "image.jpeg" if Path("image.jpeg").exists() else "image.png"
    if Path(target).exists():
        test_lp_and_ocr(target, "lp_test_result.jpg")
    else:
        print("No image.png or image.jpeg found.")
