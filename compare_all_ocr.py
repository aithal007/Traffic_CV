import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# Add LPRNet path to sys.path to resolve imports correctly
sys.path.append(os.path.join(os.getcwd(), 'license_plate_lab', 'LPRNet_Pytorch'))
from model.LPRNet import build_lprnet

# ─────────────────────────────────────────────────────────────────────────────
# 1. Custom CRNN Architecture & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
CRNN_CHARS       = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CRNN_IDX2CHAR    = {i + 1: char for i, char in enumerate(CRNN_CHARS)}
CRNN_NUM_CLASSES = len(CRNN_CHARS) + 1
CRNN_IMG_H       = 64
CRNN_IMG_W       = 200

class CRNN(nn.Module):
    def __init__(self, num_classes=CRNN_NUM_CLASSES, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (3, 2), padding=0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=False)
        self.fc  = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x).mean(dim=2).permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv)
        return nn.functional.log_softmax(self.fc(rnn_out), dim=2)

def preprocess_crnn(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    h, w = enhanced.shape
    new_w = max(1, int(w * (CRNN_IMG_H / h)))
    resized = cv2.resize(enhanced, (min(new_w, CRNN_IMG_W), CRNN_IMG_H), interpolation=cv2.INTER_LINEAR)
    if new_w < CRNN_IMG_W:
        pad_left = (CRNN_IMG_W - new_w) // 2
        pad_right = CRNN_IMG_W - new_w - pad_left
        pad_color = int(np.median(resized))
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    else:
        padded = resized
    tensor = (padded.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.FloatTensor(tensor).unsqueeze(0).unsqueeze(0)

def decode_crnn(log_probs):
    _, preds = log_probs.max(dim=1)
    decoded, prev = [], -1
    for p in preds:
        p = p.item()
        if p != 0 and p != prev and p in CRNN_IDX2CHAR:
            decoded.append(CRNN_IDX2CHAR[p])
        prev = p
    return "".join(decoded)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom LPRNet Architecture & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
LPRNET_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', '-'
               ]

def preprocess_lprnet(img, img_size=(94, 24)):
    img = cv2.resize(img, img_size)
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def decode_lprnet(preds, chars=LPRNET_CHARS):
    preds = preds.squeeze(0).detach().cpu().numpy()
    pred_labels = []
    for i in range(preds.shape[1]):
        pred_labels.append(np.argmax(preds[:, i], axis=0))
    no_repeat_blank_label = []
    pre_c = pred_labels[0]
    if pre_c != len(chars) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in pred_labels:
        if (pre_c == c) or (c == len(chars) - 1):
            if c == len(chars) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    return "".join([chars[i] for i in no_repeat_blank_label])

# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Script
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using Device: {device}\n")
    
    # Target Image
    image_path = "image.png" if os.path.exists("image.png") else "image.jpeg"
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return
    img = cv2.imread(image_path)
    print(f"Loaded image: {image_path}")
    
    # 1. Detect License Plate Crop
    detector_path = "ROLL_NUMBER/models/lp_detector.pt"
    if not os.path.exists(detector_path):
        print(f"❌ Error: Plate detector '{detector_path}' not found!")
        return
        
    print("🔍 Detecting license plate crop...")
    detector = YOLO(detector_path)
    results = detector(img, conf=0.15, verbose=False)[0]
    
    if len(results.boxes) == 0:
        print("❌ Error: No license plates detected in the image!")
        return
        
    # Crop the first detected plate
    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].tolist())
    plate_crop = img[y1:y2, x1:x2]
    print(f"🎯 Crop located at: [{x1}, {y1}, {x2}, {y2}]")
    cv2.imwrite("temp_crop.jpg", plate_crop)
    
    results_table = {}
    
    # ─── MODEL 1: Custom CRNN Model ───
    print("\n⏳ Running Model 1: CRNN...")
    crnn_weights = "crnn_best_last.pth.zip"
    if os.path.exists(crnn_weights):
        try:
            crnn_model = CRNN().to(device)
            crnn_model.load_state_dict(torch.load(crnn_weights, map_location=device))
            crnn_model.eval()
            tensor = preprocess_crnn(plate_crop).to(device)
            with torch.no_grad():
                crnn_out = crnn_model(tensor)
            crnn_pred = decode_crnn(crnn_out[:, 0, :])
            results_table["Custom CRNN"] = crnn_pred
            print(f"   👉 CRNN Result: {crnn_pred}")
        except Exception as e:
            results_table["Custom CRNN"] = f"Error: {e}"
            print(f"   ❌ CRNN Error: {e}")
    else:
        results_table["Custom CRNN"] = "Weights Not Found"
        
    # ─── MODEL 2: Custom LPRNet Model ───
    print("⏳ Running Model 2: LPRNet...")
    lprnet_weights = 'license_plate_lab/LPRNet_Pytorch/weights/best_lprnet.pth'
    if os.path.exists(lprnet_weights):
        try:
            lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(LPRNET_CHARS), dropout_rate=0.5).to(device)
            lprnet.load_state_dict(torch.load(lprnet_weights, map_location=device))
            lprnet.eval()
            lpr_tensor = preprocess_lprnet(plate_crop).to(device)
            with torch.no_grad():
                lprnet_out = lprnet(lpr_tensor)
            lprnet_pred = decode_lprnet(lprnet_out)
            results_table["Custom LPRNet"] = lprnet_pred
            print(f"   👉 LPRNet Result: {lprnet_pred}")
        except Exception as e:
            results_table["Custom LPRNet"] = f"Error: {e}"
            print(f"   ❌ LPRNet Error: {e}")
    else:
        results_table["Custom LPRNet"] = "Weights Not Found"

    # ─── MODEL 3: fast-plate-ocr (Transformer Model) ───
    print("⏳ Running Model 3: fast-plate-ocr...")
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        recognizer = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")
        fpo_res = recognizer.run(plate_crop)
        fpo_pred = ""
        if isinstance(fpo_res, (list, tuple)) and len(fpo_res) > 0:
            fpo_pred = str(getattr(fpo_res[0], "plate", fpo_res[0]))
        elif fpo_res:
            fpo_pred = str(getattr(fpo_res, "plate", fpo_res))
        results_table["fast-plate-ocr"] = fpo_pred
        print(f"   👉 fast-plate-ocr Result: {fpo_pred}")
    except Exception as e:
        results_table["fast-plate-ocr"] = f"Error: {e}"
        print(f"   ❌ fast-plate-ocr Error: {e}")

    # ─── MODEL 4: EasyOCR (English Model) ───
    print("⏳ Running Model 4: EasyOCR...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory="ROLL_NUMBER/models/easyocr")
        easy_res = reader.readtext(plate_crop)
        easy_pred = ""
        if easy_res:
            easy_pred = "".join(filter(str.isalnum, easy_res[0][1])).upper()
        results_table["EasyOCR"] = easy_pred
        print(f"   👉 EasyOCR Result: {easy_pred}")
    except Exception as e:
        results_table["EasyOCR"] = f"Error: {e}"
        print(f"   ❌ EasyOCR Error: {e}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Display Comparison Table
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("🏆 OCR MODELS COMPARISON ON THE SAME CROP 🏆")
    print("="*50)
    print(f"{'OCR Engine':<22} | {'Predicted Text':<20}")
    print("-"*50)
    for model_name, prediction in results_table.items():
        print(f"{model_name:<22} | {prediction:<20}")
    print("="*50)

if __name__ == "__main__":
    main()
