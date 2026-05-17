from ultralytics import YOLO
import cv2
import os

# 1. Path to your new model
model_path = "ROLL_NUMBER/models/bike_best.pt"
model = YOLO(model_path)

# 2. Path to your test image
test_image = "image.png"

if os.path.exists(test_image):
    print(f"Testing on image: {test_image}")
    
    # 3. Run Inference
    results = model(test_image)
    
    # 4. Save and Show Result
    res_plotted = results[0].plot()
    output_path = "bike_test_result.jpg"
    cv2.imwrite(output_path, res_plotted)
    print(f"✅ Success! Result saved to: {output_path}")
    print("Check this image to see how well your model detected the bikes!")
