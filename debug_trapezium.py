import cv2
import numpy as np
import sys
sys.path.insert(0, "ROLL_NUMBER")
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
img = cv2.imread("image.png")
persons, motos = detector._detect_objects(img)

print(f"Detected {len(persons)} persons and {len(motos)} motos.")
moto_boxes = [m[:4] for m in motos]
person_boxes = [p[:4] for p in persons]

# Draw motos and trapezium
for mb in moto_boxes:
    mx1, my1, mx2, my2 = mb
    m_w = mx2 - mx1
    m_h = my2 - my1
    m_cx = (mx1 + mx2) / 2
    
    top_y = my1 - m_h * 0.4
    top_w = m_w * 1.5
    
    pt_bl = (mx1, my2)
    pt_br = (mx2, my2)
    pt_tr = (m_cx + top_w / 2, top_y)
    pt_tl = (m_cx - top_w / 2, top_y)
    
    trapezium = np.array([pt_bl, pt_tl, pt_tr, pt_br], dtype=np.int32)
    
    cv2.rectangle(img, (int(mx1), int(my1)), (int(mx2), int(my2)), (255, 0, 0), 2)
    cv2.polylines(img, [trapezium], True, (0, 255, 255), 2)

for i, pb in enumerate(person_boxes):
    px1, py1, px2, py2 = pb
    p_cx = (px1 + px2) / 2
    p_bottom_center = (int(p_cx), int(py2 - (py2 - py1) * 0.2))
    cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)
    cv2.circle(img, p_bottom_center, 5, (0, 0, 255), -1)
    cv2.putText(img, str(i), (int(px1), int(py1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite("debug_trapezium.jpg", img)
print("Saved debug_trapezium.jpg")
