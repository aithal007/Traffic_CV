import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

# Load your custom-trained models
models_dir = "ROLL_NUMBER/models"
bike_model = YOLO(os.path.join(models_dir, "bike_best.pt"))
helmet_model = YOLO(os.path.join(models_dir, "helmet_best.pt"))
lp_model = YOLO(os.path.join(models_dir, "lp_detector.pt"))

img_path = "image.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

print(f"Image resolution: {w}x{h}")

# 1. Detect Motorcycles & Persons
results = bike_model(img, conf=0.10)[0] # Very low conf to see everything
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    label = results.names[cls]
    
    color = (255, 0, 0) if "moto" in label.lower() else (0, 255, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 2. Detect Plates
lp_results = lp_model(img, conf=0.10)[0]
for box in lp_results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"plate {conf:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("debug_all_detections.jpg", img)
print("Debug image saved to debug_all_detections.jpg")
