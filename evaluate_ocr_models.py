import cv2
import sys
import time
from pathlib import Path

# Add ROLL_NUMBER to path to import the main class
sys.path.append(str(Path("ROLL_NUMBER").absolute()))
from solution import TrafficViolationDetector

try:
    from fast_plate_ocr import LicensePlateRecognizer
except ImportError:
    print("Error: fast_plate_ocr is not installed.")
    sys.exit(1)

def evaluate_ocr_models(image_path):
    print(f"--- Evaluating Fast-Plate-OCR Models on: {image_path} ---")
    
    # 1. Initialize detector just to get the license plate bounding box
    detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # 2. Extract plate crop
    print("\nDetecting License Plate Crop...")
    lp_results = detector.lp_model(img, conf=0.05, verbose=False)
    
    plates = []
    for r in lp_results:
        for b in r.boxes:
            box = [int(v) for v in b.xyxy[0].tolist()]
            plates.append(box)

    if not plates:
        print("No plates found in the image. Cannot test OCR.")
        return

    # Just take the first plate found
    x1, y1, x2, y2 = plates[0]
    plate_crop = img[y1:y2, x1:x2]
    
    crop_path = "temp_plate_crop.jpg"
    cv2.imwrite(crop_path, plate_crop)
    print(f"Saved plate crop to {crop_path}")

    # 3. Test the different OCR models provided by the repo
    models_to_test = [
        "cct-s-v2-global-model",
        "cct-xs-v2-global-model",
        "cct-s-v1-global-model",
        "cct-xs-v1-global-model"
    ]

    print("\n--- OCR Model Results ---")
    print(f"{'Model Name':<25} | {'Predicted Text':<20} | {'Confidence':<10} | {'Inference Time (ms)'}")
    print("-" * 80)

    for model_name in models_to_test:
        try:
            # Initialize the specific model
            recognizer = LicensePlateRecognizer(model_name, device="cpu")
            
            # Run inference (warmup first)
            _ = recognizer.run(plate_crop, return_confidence=True)
            
            # Timed run
            t0 = time.time()
            result = recognizer.run(plate_crop, return_confidence=True)
            t1 = time.time()
            
            inf_time = (t1 - t0) * 1000
            
            # Extract results
            # The API returns a tuple (text, conf) or just text if return_confidence=False
            if isinstance(result[0], tuple):
                text, conf = result[0]
            else:
                text = result[0]
                conf = "N/A"
                
            print(f"{model_name:<25} | {str(text):<20} | {str(conf):<10} | {inf_time:.2f} ms")
            
        except Exception as e:
            print(f"{model_name:<25} | ERROR: {str(e)}")

if __name__ == "__main__":
    target = "image.jpeg" if Path("image.jpeg").exists() else "image.png"
    evaluate_ocr_models(target)
