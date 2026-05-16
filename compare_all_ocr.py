import cv2
import sys
import torch
import numpy as np
import time
from pathlib import Path
from fast_plate_ocr import LicensePlateRecognizer

# Setup paths for Indian_LPR
repo_path = Path("Indian_LPR").absolute()
sys.path.append(str(repo_path))
sys.path.append(str(repo_path / "src"))
sys.path.append(str(repo_path / "src" / "License_Plate_Recognition"))
from License_Plate_Recognition.model.LPRNet import build_lprnet
from License_Plate_Recognition.data.load_data import CHARS as LPR_CHARS

# Setup paths for our own detector
sys.path.append(str(Path("ROLL_NUMBER").absolute()))
from solution import TrafficViolationDetector

def compare_all(image_path):
    print(f"=== Comparing All OCR Models on: {image_path} ===")
    
    detector = TrafficViolationDetector(model_dir="ROLL_NUMBER/models")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # 1. Detect Plate
    lp_results = detector.lp_model(img, conf=0.05, verbose=False)
    if not lp_results or not lp_results[0].boxes:
        print("No plates found.")
        return

    # Process each detected plate
    for p_idx, p_box in enumerate(lp_results[0].boxes):
        box = [int(v) for v in p_box.xyxy[0].tolist()]
        plate_crop = img[box[1]:box[3], box[0]:box[2]]
        cv2.imwrite(f"plate_crop_{p_idx+1}.jpg", plate_crop)
        print(f"\n--- Plate {p_idx+1} BBox: {box} (Saved to plate_crop_{p_idx+1}.jpg) ---")
        
        results = []

        # Global Models
        global_models = [
            "cct-s-v2-global-model",
            "cct-xs-v2-global-model",
            "global-plates-mobile-vit-v2-model", # The "Old" model
            "cct-s-v1-global-model",
            "cct-xs-v1-global-model"
        ]
        
        for m_name in global_models:
            try:
                recognizer = LicensePlateRecognizer(m_name, device="cpu")
                
                # Apply raw color crop by default
                test_crop = plate_crop.copy()
                if m_name == "global-plates-mobile-vit-v2-model":
                    test_crop = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)
                
                t0 = time.time()
                # Request confidence from the recognizer
                res_obj = recognizer.run(test_crop, return_confidence=True)[0]
                inf_time = (time.time() - t0) * 1000
                
                # Calculate average confidence
                avg_conf = 0.0
                if hasattr(res_obj, "char_probs"):
                    avg_conf = np.mean(res_obj.char_probs)
                
                results.append([m_name, str(res_obj.plate), f"{avg_conf:.2f}", f"{inf_time:.1f}ms"])
            except Exception as e:
                results.append([m_name, f"Error: {e}", "N/A"])

        # B. Indian LPRNet
        try:
            lprnet = build_lprnet(lpr_max_len=18, phase=False, class_num=len(LPR_CHARS), dropout_rate=0)
            lprnet.load_state_dict(torch.load("Indian_LPR/weights/best_lprnet.pth", map_location="cpu"))
            lprnet.eval()
            
            lpr_img = cv2.resize(plate_crop, (94, 24))
            lpr_img = lpr_img.astype("float32")
            lpr_img -= 127.5
            lpr_img *= 0.0078125
            lpr_img = np.transpose(lpr_img, (2, 0, 1))
            lpr_input = torch.from_numpy(lpr_img).unsqueeze(0)
            
            t0 = time.time()
            with torch.no_grad():
                logits = lprnet(lpr_input)
                # Compute confidence from softmax
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                prebs = logits.cpu().numpy()
            inf_time = (time.time() - t0) * 1000
            
            # Simple average confidence of winning classes
            conf_score = np.mean(np.max(probs[0], axis=0))
            
            preb = prebs[0, :, :]
            preb_label = []
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat = []
            pre_c = preb_label[0]
            if pre_c != len(LPR_CHARS)-1: no_repeat.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(LPR_CHARS)-1):
                    if c == len(LPR_CHARS)-1: pre_c = c
                    continue
                no_repeat.append(c)
                pre_c = c
            lpr_res = "".join([LPR_CHARS[idx] for idx in no_repeat])
            results.append(["Indian_LPRNet", lpr_res, f"{conf_score:.2f}", f"{inf_time:.1f}ms"])
        except Exception as e:
            results.append(["Indian_LPRNet", f"Error: {e}", "N/A"])

        # Print results
        print(f"{'Model':<35} | {'Prediction':<15} | {'Conf':<6} | {'Time'}")
        print("-" * 80)
        for r in results:
            print(f"{r[0]:<35} | {r[1]:<15} | {r[2]:<6} | {r[3]}")

if __name__ == "__main__":
    compare_all("image.png")
