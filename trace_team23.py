import sys
import os
from pathlib import Path
import cv2

# Add Team23 to path so we can import the class
sys.path.insert(0, str(Path("Team23").absolute()))
import solution

def main():
    print("="*60)
    print("RUNNING DETAILED TRAPEZIUM PIPELINE TRACE ON IMAGE.PNG (TEAM23)")
    print("="*60)
    
    # Initialize the model with Team23/models
    detector = solution.TrafficViolationDetector(model_dir="Team23/models")
    img = cv2.imread("image.png")
    if img is None:
        print("Error: Could not load image.png")
        return
    h, w = img.shape[:2]
    print(f"Loaded image.png of size {w}x{h}\n")
    
    # 1. Detect Motorcycles
    bike_res = detector.bike_detector(img, conf=0.15, verbose=False)
    all_bikes = []
    for r in bike_res:
        for b in r.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            all_bikes.append([bx1, by1, bx2, by2, conf])
            
    print(f"🏍️ Motorcycle Detector located {len(all_bikes)} motorcycle(s):")
    for i, b in enumerate(all_bikes):
        print(f"   Motorcycle {i+1}: {b[:4]} (confidence: {b[4]:.2f})")
        
    # 2. Detect Heads/Helmets
    head_res = detector.helmet_model(img, conf=0.15, verbose=False)
    all_heads = []
    for r in head_res:
        for b in r.boxes:
            hx1, hy1, hx2, hy2 = [int(v) for v in b.xyxy[0].tolist()]
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            label = "No Helmet" if cls_id == detector.no_helmet_id else "Helmet"
            all_heads.append([hx1, hy1, hx2, hy2, cls_id, conf, label])
            
    print(f"\n👷 Helmet/Head Detector located {len(all_heads)} head(s):")
    for i, h_info in enumerate(all_heads):
        print(f"   Head {i+1}: {h_info[:4]} | Class: {h_info[6]} (confidence: {h_info[5]:.2f})")
        
    # 3. Process each motorcycle's trapezium and run association
    for i, b in enumerate(all_bikes):
        bx1, by1, bx2, by2, bconf = b
        print(f"\n--- Processing Motorcycle {i+1} Association ---")
        
        # Calculate mathematical exclusive trapezium
        trap = solution._get_exclusive_trapezium((bx1, by1, bx2, by2), all_bikes, w, h)
        print(f"   📐 Exclusive Trapezium Points: {trap.tolist()}")
        
        # Check which heads/riders fall into this trapezium
        associated_heads = []
        for h_info in all_heads:
            hx1, hy1, hx2, hy2, cls_id, conf, label = h_info
            cx, cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
            if solution._point_in_polygon((cx, cy), trap):
                associated_heads.append(h_info)
                
        print(f"   👥 Associated {len(associated_heads)} rider(s) within the exclusive trapezium:")
        for ah in associated_heads:
            print(f"      - Rider Head at {ah[:4]} | {ah[6]} (conf: {ah[5]:.2f})")
            
        # 4. Get Plate Candidates for this bike
        print("   🔍 Searching for associated plate candidates in motorcycle search area:")
        candidates = detector._get_plate_candidates(img, (bx1, by1, bx2, by2))
        if not candidates:
            print("      ❌ No plate candidates found in the search area.")
        for ci, cand in enumerate(candidates):
            ocr_texts = detector._run_plate_ocr(cand["crop"])
            print(f"      Candidate {ci+1} ({cand['source']}):")
            print(f"         Box: {cand['box']}")
            print(f"         Raw OCR Reads: {ocr_texts}")
            print(f"         Merged & Normalized Result: {solution._smart_plate_merge(ocr_texts)}")
            
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
