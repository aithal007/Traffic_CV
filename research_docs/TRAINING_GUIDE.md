# Training Guide: Traffic Violation Detection Models

## Overview

This guide walks through fine-tuning YOLO11n models for:
1. **Motorcycle Detection** (Stage 1) - Detect motorcycles and riders
2. **Helmet Detection** (Stage 2) - Classify helmet presence
3. **License Plate Detection** (Stage 3) - Detect plates for OCR

---

## Quick Start (< 2 hours with sample data)

### 1. Prepare Your Data

**Download sample datasets** (or use your own):

```bash
# Option A: Use Roboflow (recommended)
pip install roboflow
python
>>> from roboflow import Roboflow
>>> rf = Roboflow(api_key="YOUR_API_KEY")
>>> project = rf.workspace("workspace-name").project("project-name")
>>> dataset = project.version(1).download("yolov8")

# Option B: Download from CCPD (Chinese City Parking Dataset)
# Visit: https://github.com/detectRecog/CCPD
# Download CCPD-Base (~3.5GB) or CCPD-Challenge

# Option C: Use IITH Helmet Dataset
# Contact: IITH for academic access
# Dataset: https://github.com/iith-cvs-lab/HELMET-dataset
```

### 2. Organize Dataset in YOLO Format

```
dataset/
├── images/
│   ├── train/     ← 80% of images
│   ├── val/       ← 10% of images
│   └── test/      ← 10% of images (optional)
└── labels/        ← Corresponding .txt annotations
    ├── train/
    ├── val/
    └── test/
```

**YOLO Label Format** (one .txt per image):
```
<class_id> <x_center> <y_center> <width> <height>  [normalized 0-1]
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.1 0.15
```

**Validate annotations**:
```bash
python -c "
import os
for root, dirs, files in os.walk('dataset/'):
    imgs = sum(1 for f in os.listdir('dataset/images/train') if f.endswith(('.jpg','.png')))
    labels = sum(1 for f in os.listdir('dataset/labels/train') if f.endswith('.txt'))
    if imgs != labels:
        print(f'Mismatch: {imgs} images vs {labels} labels')
"
```

### 3. Create data.yaml

```yaml
# dataset/data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 2  # Number of classes
names: 
  0: motorcycle
  1: rider

# Optional: For helmet detection
# nc: 2
# names:
#   0: helmet
#   1: no_helmet
```

### 4. Start Training

**Stage 1: Motorcycle Detection (Most Important)**
```bash
cd ROLL_NUMBER
python train.py --stage motorcycle \
    --data ../dataset/data.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0  # GPU device ID, use 'cpu' if no GPU
```

**Expected Output**:
```
Starting training...
      epoch    train/box_loss  train/cls_loss  ...  val/mAP50  val/mAP50-95
          1         2.456      1.234       ...     0.123      0.089
          2         2.210      1.089       ...     0.234      0.167
        ...
        100         0.234      0.089       ...     0.856      0.742

Best mAP50: 0.856 at epoch 85
Model saved to: training_output/motorcycle_detector/run_20260512_143022/weights/best.pt
```

---

## Full Multi-Stage Training (4-6 hours with GPU)

### Prepare 3 Datasets

You need 3 separate datasets with different class definitions:

**Dataset 1: Motorcycle Detection**
- Classes: motorcycle, rider
- ~5000 images recommended

**Dataset 2: Helmet Detection**  
- Classes: helmet, no_helmet
- Crop rider regions for focused training
- ~3000 images recommended

**Dataset 3: License Plate Detection**
- Classes: plate
- Crop motorcycle bottom regions
- ~2000 images recommended

### Train All Stages

```bash
cd ROLL_NUMBER

# Full pipeline training
python train.py --pipeline full \
    --motorcycle-data ../motorcycle_dataset/data.yaml \
    --helmet-data ../helmet_dataset/data.yaml \
    --plate-data ../plate_dataset/data.yaml \
    --epochs 100 80 60 \
    --device 0
```

**Timeline**:
- Motorcycle (100 epochs): ~1.5-2 hours (GPU)
- Helmet (80 epochs): ~1-1.5 hours (GPU)
- Plate (60 epochs): ~45-60 minutes (GPU)
- Total: ~4-5 hours

**Without GPU (CPU only)**: ~20-40 hours (not recommended)

---

## Dataset Recommendations

### Free/Public Datasets

| Dataset | Classes | Images | License | Link |
|---------|---------|--------|---------|------|
| CCPD | plate | 290K | CC BY-NC 3.0 | [GitHub](https://github.com/detectRecog/CCPD) |
| ELP (ALPR) | plate | 100K+ | OpenALPR | [Website](https://opensource.com/article/20/11/open-source-license-plate-recognition) |
| BIT-Vehicle | motorcycle | 9K+ | Research | [GitHub](https://github.com/BitVehicle/BIT-Vehicle) |
| IITH Helmet | helmet | 6K+ | Academic | [GitHub](https://github.com/iith-cvs-lab/HELMET-dataset) |

### Commercial/Academic Datasets

| Dataset | Details | Contact |
|---------|---------|---------|
| RideSafe-400 | 354K R-M annotations | Request from authors |
| DataCluster HSRP | 15K Indian plates | Commercial/Research |
| TensorFlow COCO | General objects | Free download |

### Build Your Own Dataset

```bash
# 1. Collect video from traffic cameras
# 2. Extract frames: ffmpeg -i video.mp4 -r 1 frame_%05d.jpg

# 3. Annotate using Roboflow or LabelImg
pip install roboflow labelimg

# 4. Export in YOLO format (Roboflow auto-converts)

# 5. Split into train/val/test
python -c "
import os, shutil, random
from pathlib import Path
Path('dataset/images').mkdir(parents=True)
# ... split logic ...
"
```

---

## Training Configuration Details

### Hyperparameters (Tuned for Traffic Violation Detection)

**Stage 1: Motorcycle Detection** (Conservative)
```yaml
lr0: 0.01           # Initial learning rate
lrf: 0.01           # Final learning rate (10% of initial)
momentum: 0.937
weight_decay: 0.0005
degrees: 10         # Rotation augmentation
translate: 0.1      # Spatial translation
scale: 0.5          # Scale augmentation (0.5-1.5x)
hsv_h: 0.015        # HSV Hue variation
hsv_s: 0.7          # HSV Saturation variation
hsv_v: 0.4          # HSV Value variation
flipud: 0.5         # Vertical flip probability
fliplr: 0.5         # Horizontal flip probability
```

**Stage 2: Helmet Detection** (Aggressive)
```yaml
degrees: 15         # More rotation (helmets at angles)
translate: 0.15
scale: 0.6
hsv_h: 0.02         # More color variation
hsv_s: 0.8
hsv_v: 0.5
mosaic: 1.0         # Mosaic augmentation on
```

**Stage 3: License Plate Detection** (Conservative for Text)
```yaml
degrees: 5          # Minimal rotation (preserves readability)
translate: 0.05
scale: 0.3
hsv_h: 0.01         # Minimal color change
hsv_s: 0.5
hsv_v: 0.3
flipud: 0.0         # No vertical flip (text orientation)
```

### Loss Functions

```
Total Loss = λ1 * Box Loss + λ2 * Confidence Loss + λ3 * Class Loss

Box Loss (GIoU):        Penalizes bounding box regression errors
Confidence Loss (BCE):  Penalizes objectness prediction
Class Loss (BCE):       Penalizes class prediction
```

---

## Monitoring Training

### Real-time Metrics Dashboard

```bash
# TensorBoard visualization (if TensorBoard logging enabled)
tensorboard --logdir training_output

# Access at: http://localhost:6006
```

### Key Metrics to Monitor

**During Training**:
- `train/box_loss`: Should decrease smoothly
- `train/cls_loss`: Class loss trend
- `val/mAP50`: Validation mean average precision

**Ideal Progression**:
```
Epoch 1-20:   Rapid loss decrease (steep curve)
Epoch 20-60:  Moderate loss decrease (gentle curve)
Epoch 60-100: Plateau (little improvement, possible overfitting)
              → Early stopping triggered
```

### Early Stopping

Training automatically stops when validation mAP doesn't improve for:
- Motorcycle: 20 epochs
- Helmet: 15 epochs
- Plate: 12 epochs

```bash
# Manual early stopping: Ctrl+C
# Last checkpoint saved automatically
```

---

## Validation & Testing

### Validate After Training

```bash
# Single model validation
python train.py --validate training_output/motorcycle_detector/run_*/weights/best.pt \
    --data ../motorcycle_dataset/data.yaml

# Expected output
# mAP50: 0.856
# mAP50-95: 0.742
# Precision: 0.873
# Recall: 0.821
```

### Test on Custom Images

```python
from ultralytics import YOLO

model = YOLO('training_output/motorcycle_detector/run_*/weights/best.pt')
results = model.predict('test_image.jpg', conf=0.5)
```

---

## Model Export & Deployment

### Export Trained Models

```bash
# Export to ONNX (recommended for deployment)
python train.py --export training_output/motorcycle_detector/run_*/weights/best.pt \
    --format onnx

# Other formats: torchscript, tflite, pb, engine (TensorRT), etc.

# Result: best.onnx (~5 MB)
```

### Use Exported Model

```python
from ultralytics import YOLO

# Load ONNX model
model = YOLO('best.onnx')
results = model.predict('image.jpg')
```

---

## Troubleshooting

### Problem: Low Validation Accuracy

**Symptoms**: mAP50 < 0.5, training loss decreases but validation stagnates

**Solutions**:
1. **Check dataset**:
   - Verify label alignment (check 10 random annotations)
   - Ensure balanced class distribution
   - Remove corrupted images

2. **Increase training data**:
   - Add more varied examples (different angles, lighting, weather)
   - Use data augmentation more aggressively

3. **Adjust hyperparameters**:
   ```bash
   # Try lower learning rate
   python train.py --stage motorcycle --lr0 0.005 --lrf 0.005
   
   # Try larger batch size (if GPU memory allows)
   python train.py --stage motorcycle --batch 32
   ```

### Problem: Training Too Slow

**Symptoms**: 1+ hour for 10 epochs

**Solutions**:
1. **Use GPU**:
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Force GPU usage
   python train.py --device 0  # Device 0 = first GPU
   ```

2. **Reduce image size**:
   ```bash
   # Helmet & Plate: smaller images = faster training
   python train.py --img 416  # Default 640
   ```

3. **Reduce batch size** (if out of memory):
   ```bash
   python train.py --batch 8
   ```

### Problem: Out of Memory Error

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
1. **Reduce batch size**:
   ```bash
   python train.py --batch 8  # Start with small value
   ```

2. **Reduce image size**:
   ```bash
   python train.py --img 416
   ```

3. **Clear GPU cache**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

4. **Use CPU** (slow but works):
   ```bash
   python train.py --device cpu
   ```

### Problem: Model Not Converging

**Symptoms**: Loss not decreasing, stuck at random values

**Solutions**:
1. **Verify data.yaml paths** are absolute or relative to training script
2. **Check label format**: Each label line should be `<class> <x> <y> <w> <h>`
3. **Validate class count**: `nc` in data.yaml must match actual classes
4. **Reset weights**: Delete checkpoint, retrain from scratch

---

## Best Practices

### Dataset Preparation ✅
- [ ] Split: 80% train, 10% val, 10% test (or 85/8/7)
- [ ] Balance: Similar number of samples per class
- [ ] Diversity: Various angles, lighting, weather, distances
- [ ] Quality: Remove blurry, corrupted, mislabeled images
- [ ] Augmentation: Use YOLO built-in (don't pre-augment)

### Training ✅
- [ ] Start with pretrained weights (transfer learning)
- [ ] Monitor validation metrics (don't ovfit)
- [ ] Save checkpoints every N epochs
- [ ] Use learning rate scheduling (decrease over time)
- [ ] Stop early if validation metric plateaus

### Validation ✅
- [ ] Validate on held-out test set (unseen during training)
- [ ] Report: mAP50, mAP50-95, Precision, Recall, F1
- [ ] Analyze failure cases (false positives/negatives)
- [ ] Compare to baseline (pretrained model)

### Deployment ✅
- [ ] Export to ONNX for cross-platform compatibility
- [ ] Test on production hardware (GPU/edge device)
- [ ] Monitor inference latency (<600 ms target)
- [ ] Track model versioning and metrics

---

## Advanced: Multi-GPU Training

```bash
# Train on multiple GPUs
python train.py --device 0,1,2,3 --batch 64

# Distributed training across machines
# (requires torch.distributed setup)
```

---

## Next Steps

1. **Collect/Download Dataset** (~2-4 hours)
2. **Annotate in YOLO Format** (~4-8 hours for custom data)
3. **Create data.yaml** (~5 minutes)
4. **Run Training** (Stage 1: 1-2 hours GPU, then stages 2-3)
5. **Validate & Export** (~30 minutes)
6. **Deploy Models** to solution.py (~1 hour)

---

## Training Timeline Example

```
Day 1:
  ├─ 09:00 - Download dataset (CCPD + IITH)
  ├─ 10:00 - Organize in YOLO format
  ├─ 10:30 - Create data.yaml
  └─ 11:00 - Start Motorcycle Detector training

Day 2:
  ├─ 08:00 - Training completed (12+ hours)
  ├─ 08:30 - Validate model (mAP check)
  ├─ 09:00 - Start Helmet Detector training
  └─ 18:00 - Training completed

Day 3:
  ├─ 08:00 - Validate Helmet model
  ├─ 09:00 - Start Plate Detector training
  └─ 14:00 - Training completed

Day 4:
  ├─ 08:00 - Validate all models
  ├─ 09:00 - Export to ONNX
  ├─ 10:00 - Update solution.py with trained models
  └─ 12:00 - Deploy & test
```

---

## Resources

- **Ultralytics YOLO Docs**: https://docs.ultralytics.com
- **Roboflow Dataset Docs**: https://roboflow.com
- **YOLO Training Logs**: Training output contains detailed loss curves
- **Community Help**: Ultralytics GitHub Issues & Discussions

---

**Ready to train? Start with Stage 1! 🚀**
