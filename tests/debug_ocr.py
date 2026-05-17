
import cv2
import os
import sys
from pathlib import Path

# Add ROLL_NUMBER to path
sys.path.append(os.path.join(os.getcwd(), "ROLL_NUMBER"))
from solution import TrafficViolationDetector, _safe_crop, _preprocess_plate

def debug_ocr(image_path):
    detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return

    # Run bike detection
    res = detector.bike_detector(img, conf=0.12, verbose=False)
    for r in res:
        for b in r.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            b_crop = _safe_crop(img, (bx1, by1, bx2, by2))
            
            # Detect plates
            lp_res = detector.lp_model(b_crop, conf=0.10, verbose=False)
            for lr in lp_res:
                for lb in lr.boxes:
                    lx1, ly1, lx2, ly2 = [int(v) for v in lb.xyxy[0].tolist()]
                    plate_crop = _safe_crop(b_crop, (lx1, ly1, lx2, ly2))
                    if plate_crop is not None:
                        print(f"Plate found! Size: {plate_crop.shape}")
                        
                        # Test EasyOCR
                        if detector.ocr_reader:
                            # 1. Raw
                            raw_res = detector.ocr_reader.readtext(plate_crop)
                            print(f"Raw OCR: {raw_res}")
                            
                            # 2. Resized Raw
                            h, w = plate_crop.shape[:2]
                            resized = cv2.resize(plate_crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                            res_res = detector.ocr_reader.readtext(resized)
                            print(f"Resized OCR: {res_res}")
                            
                            # 3. Preprocessed
                            for i, ver in enumerate(_preprocess_plate(plate_crop)):
                                ver_res = detector.ocr_reader.readtext(ver)
                                print(f"Ver {i} OCR: {ver_res}")

if __name__ == "__main__":
    debug_ocr("image.png")
