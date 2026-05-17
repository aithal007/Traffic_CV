import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path("Group23").absolute()))
import solution

def main():
    detector = solution.TrafficViolationDetector(model_dir="Group23/models")
    output = detector.predict("image.png")
    print("\n--- FINAL JSON OUTPUT ---")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
