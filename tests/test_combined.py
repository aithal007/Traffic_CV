from ultralytics import YOLO
import cv2
import os
import numpy as np

# 1. Load both models
bike_model = YOLO("ROLL_NUMBER/models/bike_best.pt")
helmet_model = YOLO("ROLL_NUMBER/models/helmet_best.pt")

# 2. Path to test image
test_image = "image.png"

if os.path.exists(test_image):
    print(f"Testing on image: {test_image}")
    img = cv2.imread(test_image)
    
    # 1. Detect Motorcycles
    bike_results = bike_model(test_image)[0]
    
    # 2. For each bike found, look for helmets
    for i, bike_box in enumerate(bike_results.boxes):
        x1, y1, x2, y2 = bike_box.xyxy[0].cpu().numpy().astype(int)
        
        # Crop the bike (with a little extra space for the head)
        crop_y1 = max(0, y1 - 50)
        bike_crop = img[crop_y1:y2, x1:x2]
        
        if bike_crop.size == 0: continue
            
        # 3. Detect Helmet on the CROP (much easier!)
        helmet_results = helmet_model(bike_crop, conf=0.1)[0] # Lower conf for testing
        
        # Draw the bike box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img, "motorcycle", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw the helmet results back onto the main image
        for h_box in helmet_results.boxes:
            hx1, hy1, hx2, hy2 = h_box.xyxy[0].cpu().numpy().astype(int)
            label = helmet_results.names[int(h_box.cls[0])]
            conf = float(h_box.conf[0])
            
            # Map crop coordinates back to main image
            final_x1, final_y1 = x1 + hx1, crop_y1 + hy1
            final_x2, final_y2 = x1 + hx2, crop_y1 + hy2
            
            color = (0, 255, 0) if "with" in label.lower() else (0, 0, 255)
            cv2.rectangle(img, (final_x1, final_y1), (final_x2, final_y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (final_x1, final_y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. Save result
    output_path = "zoomed_test_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"✅ Success! Zoomed result saved to: {output_path}")
else:
    print(f"Error: {test_image} not found!")
