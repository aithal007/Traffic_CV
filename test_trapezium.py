import sys
import cv2
import json
import time
from pathlib import Path

# Add ROLL_NUMBER to path so we can import the class
sys.path.append(str(Path("ROLL_NUMBER").absolute()))
from solution import TrafficViolationDetector, _get_exclusive_trapezium, _safe_crop, _smart_plate_merge

def test_and_visualize(img_path, out_path):
    # Initialize the model
    detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
    
    # Warm-up run (PyTorch allocates memory on the first forward pass)
    print("Warming up models...")
    _ = detector.predict(img_path)
    
    # Run the prediction with timing (true inference time)
    t0 = time.time()
    result = detector.predict(img_path)
    t1 = time.time()
    
    inference_time = (t1 - t0) * 1000 # in ms
    print(f"\n>>> TRUE PIPELINE INFERENCE TIME (After Warmup): {inference_time:.2f} ms")
    
    # Write JSON results to a file and print them
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n--- EXACT JSON OUTPUT ---")
    print(json.dumps(result, indent=2))
    print("-------------------------\n")

    # Now let's draw everything to prove the math works!
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return
    h, w = img.shape[:2]

    violations = result.get("violations", [])

    # Collect all bikes first
    bike_res = detector.bike_detector(img, conf=0.10, verbose=False)
    all_bikes_viz = []
    for r in bike_res:
        for b in r.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            all_bikes_viz.append([bx1, by1, bx2, by2, float(b.conf[0])])

    # Draw Bikes & ROIs
    for b_idx, b in enumerate(all_bikes_viz):
        bx1, by1, bx2, by2, bconf = b
        bw, bh = bx2 - bx1, by2 - by1
        
        # Determine if this bike is in the violation list (rough bbox match)
        is_violating = False
        for v in violations:
            # Note: predict() doesn't return bboxes anymore in the clean version, 
            # so we'd have to re-run or use internal logic. 
            # But for viz, let's just mark bikes that have riders/violations.
            pass

        cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        cv2.putText(img, f"Bike {bconf:.2f}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw Actual Mathematical Exclusive Trapezium (Cyan)
        trap_pts = _get_exclusive_trapezium((bx1, by1, bx2, by2), all_bikes_viz, w, h)
        cv2.polylines(img, [trap_pts], isClosed=True, color=(255, 255, 0), thickness=1)

    # Draw Heads
    head_res = detector.helmet_model(img, conf=0.15, verbose=False)
    for r in head_res:
        for b in r.boxes:
            hx1, hy1, hx2, hy2 = [int(v) for v in b.xyxy[0].tolist()]
            h_cls = int(b.cls[0])
            color = (0, 0, 255) if h_cls == detector.no_helmet_id else (0, 255, 0)
            label = "No Helmet" if h_cls == detector.no_helmet_id else "Helmet"
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), color, 2)
            cx, cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
            cv2.circle(img, (cx, cy), 4, color, -1)

    # Global LP Detection Debug
    lp_res = detector.lp_model(img, conf=0.15, verbose=False)
    for r in lp_res:
        for b in r.boxes:
            lx1, ly1, lx2, ly2 = [int(v) for v in b.xyxy[0].tolist()]
            cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (255, 0, 255), 2)
            
            # Run OCR on this plate
            plate_crop = _safe_crop(img, (lx1, ly1, lx2, ly2))
            ocr_texts = detector._run_plate_ocr(plate_crop)
            label = "".join(ocr_texts) if ocr_texts else "???"
            cv2.putText(img, f"LP: {label}", (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imwrite(out_path, img)
    print(f"Visual proof saved to {out_path}!")

if __name__ == "__main__":
    target = "image.jpeg" if Path("image.jpeg").exists() else "image.png"
    if Path(target).exists():
        test_and_visualize(target, "annotated_trapezium.jpg")
    else:
        print("No image found.")
