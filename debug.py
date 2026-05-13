import cv2
import sys
sys.path.insert(0, "ROLL_NUMBER")
from solution import TrafficViolationDetector
detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
img = cv2.imread("image.png")
persons, motos = detector._detect_objects(img)
print(f"Detected {len(persons)} persons and {len(motos)} motos.")

for i, p in enumerate(persons):
    status = detector._classify_helmet(img, p[:4])
    print(f"Person {i} bbox {p[:4]} conf {p[4]:.2f} -> Helmet status: {status}")

from solution import _associate_riders_to_motorcycles
person_boxes = [p[:4] for p in persons]
moto_boxes = [m[:4] for m in motos]
print(f"Moto boxes: {moto_boxes}")
assignments = _associate_riders_to_motorcycles(person_boxes, moto_boxes)
for mi, a in enumerate(assignments):
    print(f"Moto {mi} assigned persons: {a}")
