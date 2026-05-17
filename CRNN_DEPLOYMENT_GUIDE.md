# 🚀 CRNN License Plate OCR — Deployment & Inference Guide

This guide contains everything your collaborator needs to successfully load, preprocess, and run inference using the custom-trained PyTorch CRNN model (`crnn_best.pth`).

---

## 📦 1. Files to Send to Your Friend

You only need to send **two files** for standalone OCR inference:
1. **`models/crnn_best.pth`**: The trained PyTorch weights file (approx. 36 MB).
2. **`crnn_inference.py`** (provided below): A standalone Python script containing the model architecture, exact preprocessing logic, and decoding functions.

---

## ⚠️ 2. CRITICAL Preprocessing Rules (Must Follow!)

The CRNN is a neural network trained on continuous grayscale gradients. **DO NOT apply destructive legacy thresholding (like Otsu's binarization, Canny edge detection, or heavy morphological dilation/erosion)**. Doing so destroys character edges and causes severe hallucination.

### The 3 Mandatory Preprocessing Steps:
1. **Grayscale Conversion & CLAHE:** Convert the cropped BGR plate image to grayscale. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize shadows and highlights without destroying character geometry.
2. **Aspect-Ratio Preserving Padding (200×64):** 
   - The model expects an input tensor of exactly `(1, 64, 200)` (Channels×Height×Width).
   - **DO NOT use raw `cv2.resize(img, (200, 64))`**. Direct resizing stretches square/tall plates horizontally, distorting characters (`0` becomes `D`, `1` becomes `7`).
   - **Correct Method:** Scale the image height to `64` while maintaining aspect ratio. If the resulting width is less than `200`, pad the left and right sides with the median background pixel value.
3. **Standard Normalization:** Scale pixel values from `[0, 255]` to `[-1.0, 1.0]` using `(img / 255.0 - 0.5) / 0.5`.

---

## 💻 3. Standalone Inference Script (`crnn_inference.py`)

Your friend can save the code below as `crnn_inference.py` and run it directly on any cropped license plate image.

```python
import cv2
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & Vocab
# ─────────────────────────────────────────────────────────────────────────────
CHARS       = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
IDX2CHAR    = {i + 1: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 36 chars + 1 CTC Blank token (index 0)
IMG_H       = 64
IMG_W       = 200

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    """CNN backbone → BiLSTM → CTC classifier."""
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 32x100
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 16x50
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 8x50
            
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 4x50
            
            nn.Conv2d(512, 512, (3, 2), padding=0), nn.BatchNorm2d(512), nn.ReLU(True), # 1x49
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=False)
        self.fc  = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x).mean(dim=2).permute(2, 0, 1)  # (Width, Batch, 512)
        rnn_out, _ = self.rnn(conv)
        return nn.functional.log_softmax(self.fc(rnn_out), dim=2)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Mandatory Preprocessing & Decoding
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_plate(crop_bgr):
    """Applies CLAHE, aspect-preserving padding, and normalization."""
    # 1. Grayscale & CLAHE
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    
    # 2. Aspect-Ratio Preserving Padding
    h, w = enhanced.shape
    new_w = max(1, int(w * (IMG_H / h)))
    resized = cv2.resize(enhanced, (min(new_w, IMG_W), IMG_H), interpolation=cv2.INTER_LINEAR)
    
    if new_w < IMG_W:
        pad_left = (IMG_W - new_w) // 2
        pad_right = IMG_W - new_w - pad_left
        pad_color = int(np.median(resized))  # Median background color
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    else:
        padded = resized

    # 3. Normalization [-1.0, 1.0]
    tensor = (padded.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.FloatTensor(tensor).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 200)

def ctc_decode(log_probs):
    """Greedy CTC decoding to string."""
    _, preds = log_probs.max(dim=1)
    decoded, prev = [], -1
    for p in preds:
        p = p.item()
        if p != 0 and p != prev and p in IDX2CHAR:
            decoded.append(IDX2CHAR[p])
        prev = p
    return "".join(decoded)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Execution Example
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = CRNN().to(device)
    model.load_state_dict(torch.load("crnn_best.pth", map_location=device, weights_only=True))
    model.eval()
    
    # Load Image (Replace with actual cropped plate image path)
    img_path = "sample_plate_crop.jpg"
    image = cv2.imread(img_path)
    
    if image is not None:
        tensor = preprocess_plate(image).to(device)
        with torch.no_grad():
            output = model(tensor)  # (T, 1, NUM_CLASSES)
        
        prediction = ctc_decode(output[:, 0, :])
        print(f"OCR Prediction for {img_path}: {prediction}")
    else:
        print(f"Error: Could not load image {img_path}")
```

---

## 🛠️ 4. Environment & Dependencies

Your friend will need the following standard Python packages installed:
```bash
pip install torch torchvision torchaudio opencv-python numpy
```
*(GPU/CUDA is optional; the script automatically falls back to CPU if CUDA is unavailable).*
