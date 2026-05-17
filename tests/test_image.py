import cv2
import json
import os
import sys

# Add ROLL_NUMBER to path so we can import solution
sys.path.append(os.path.join(os.getcwd(), "ROLL_NUMBER"))
from solution import TrafficViolationDetector

def run_test(image_path):
    # Enable debug logs
    os.environ["TVD_DEBUG"] = "1"
    
    # Initialize detector
    detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
    
    # Process image
    print(f"Processing {image_path}...")
    results = detector.predict(image_path)
    
    # ── Visualization ──
    img = cv2.imread(image_path)
    # We'll run a quick internal pass to get boxes for drawing
    # (In a real scenario, predict() would return boxes, but we'll re-detect for the debug view)
    
    # Drawing logic
    violations = results.get("violations", [])
    print(f"\nFound {len(violations)} violations.")
    
    # Save the JSON results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    print("Full pipeline check complete.")
    return results

if __name__ == "__main__":
    test_img = "image.png"
    if os.path.exists(test_img):
        run_test(test_img)
    else:
        print(f"Error: {test_img} not found.")
