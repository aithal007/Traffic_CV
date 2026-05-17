import sys
import os
import json
import time

# Add Team23 directory to sys.path
sys.path.insert(0, os.path.abspath('Team23'))

from solution import TrafficViolationDetector

def main():
    print("Testing Team23 solution...")
    start_time = time.time()
    
    # Initialize detector using default path resolution (relocatable)
    detector = TrafficViolationDetector()
    print(f"Detector initialized in {time.time() - start_time:.2f} seconds.")
    
    image_path = "image.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
        
    print(f"Running prediction on {image_path}...")
    pred_start = time.time()
    try:
        output = detector.predict(image_path)
        print(f"Prediction completed in {time.time() - pred_start:.2f} seconds.")
        print("\n--- VIOLATIONS JSON ---")
        print(json.dumps(output, indent=2))
        print("-----------------------")
    except Exception as e:
        print("An error occurred during prediction:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
