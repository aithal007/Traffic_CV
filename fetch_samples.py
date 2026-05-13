import urllib.request
import os
from pathlib import Path

# Create a folder for test images
test_dir = Path("test_images")
test_dir.mkdir(exist_ok=True)

# Sample image URLs (Publicly available images for research/testing)
# Note: These are representative examples
images = {
    "triple_riding_1.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", # Placeholder for actual traffic URL logic
    "helmet_violation_1.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg", # Placeholder
}

# I will fetch some more specific traffic violation images if possible, 
# but for now let's set up the structure.

def download_samples():
    print("Downloading sample real-world images...")
    # Using some more direct traffic related images from known public datasets/repos
    samples = [
        ("traffic_1.jpg", "https://github.com/ultralytics/assets/releases/download/v0.0.0/traffic.jpg"),
        ("motorcycle_1.jpg", "https://images.pexels.com/photos/2116489/pexels-photo-2116489.jpeg"), # A generic motorcycle photo
    ]
    
    for name, url in samples:
        dest = test_dir / name
        if not dest.exists():
            print(f"  Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, str(dest))
                print(f"  [OK] Saved to {dest}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
        else:
            print(f"  [SKIP] {name} already exists")

if __name__ == "__main__":
    download_samples()
