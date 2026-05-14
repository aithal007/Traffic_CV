# 🚀 Quick Start: Train Your First Model (< 1 Hour)

**Goal**: Start training in 5 minutes with sample data, then scale to production

---

## Step 1: Install Training Dependencies (5 min)

```bash
cd ROLL_NUMBER
pip install -r requirements.txt
```

### Verify Installation
```bash
# Check CUDA GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
# Output: GPU Available: True (or False for CPU-only)
```

---

## Step 2: Get Sample Dataset (10 min)

### Option A: Quick Sample (Fastest - Download ~100MB)
```bash
# Create sample dataset structure
python train_quick.py --create-sample

# This creates:
# sample_dataset/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
```

### Option B: Real Dataset (Recommended - 1-2 hours)

**Download CCPD Dataset** (Chinese license plates):
```bash
# 1. Visit: https://github.com/detectRecog/CCPD
# 2. Download CCPD-Base (3.5GB)
# 3. Extract and organize:

python -c "
from pathlib import Path
import shutil

# Organize CCPD for YOLO format
# Expected: images in images/, labels in labels/
# See TRAINING_GUIDE.md for details
"
```

Or use **Roboflow** (easiest):
```bash
pip install roboflow

python -c "
from roboflow import Roboflow

# Get free API key from roboflow.com
rf = Roboflow(api_key='YOUR_FREE_API_KEY')
project = rf.workspace('motorcycle-detection').project('traffic-violations')
dataset = project.version(1).download('yolov8')
"
```

---

## Step 3: Prepare Your Data (5 min)

### File Structure
```
motorcycle_dataset/
├── images/
│   ├── train/      (80% of images)
│   ├── val/        (10% of images)
│   └── test/       (10% of images)
├── labels/         (same structure with .txt files)
└── data.yaml       ← Create this file
```

### Create data.yaml
```yaml
# motorcycle_dataset/data.yaml
path: C:\Users\Lenovo\Documents\cv_project\motorcycle_dataset
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: motorcycle
  1: rider
```

### Verify Data
```bash
# Check alignment
python -c "
import os

train_imgs = len(os.listdir('motorcycle_dataset/images/train'))
train_labels = len(os.listdir('motorcycle_dataset/labels/train'))

print(f'Images: {train_imgs}')
print(f'Labels: {train_labels}')

if train_imgs == train_labels:
    print('✓ Dataset OK')
else:
    print('✗ Mismatch in data')
"
```

---

## Step 4: Start Training (5-60 min depending on dataset size)

### Quick Start (5 min with sample data)
```bash
cd ROLL_NUMBER

# Train motorcycle detector
python train_quick.py \
    --stage motorcycle \
    --data ../sample_dataset/data.yaml \
    --epochs 10 \
    --batch 8
```

**Output**:
```
Starting training...
      epoch    train/box_loss  val/mAP50
          1         2.456      0.123
          2         2.210      0.234
          ...
         10         1.234      0.567

✓ Training completed successfully!
Best model: training_output/motorcycle_detector/run_20260512_143022/weights/best.pt
```

### Full Training (45 min - 2 hours with real data)
```bash
# Real dataset with GPU
python train_quick.py \
    --stage motorcycle \
    --data ../motorcycle_dataset/data.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0

# Monitor in another terminal
tensorboard --logdir training_output
# Open browser: http://localhost:6006
```

---

## Step 5: Validate Your Model (2 min)

```bash
# Find your best model path
ls training_output/motorcycle_detector/run_*/weights/best.pt

# Validate
python train_quick.py \
    --validate training_output/motorcycle_detector/run_20260512_143022/weights/best.pt \
    --data ../motorcycle_dataset/data.yaml
```

**Expected Output**:
```
mAP50: 0.856
mAP50-95: 0.742
Precision: 0.873
Recall: 0.821
✓ Validation completed!
```

---

## Step 6: Use Your Trained Model

### Export to ONNX
```bash
python train_quick.py \
    --export training_output/motorcycle_detector/run_20260512_143022/weights/best.pt \
    --format onnx

# Creates: training_output/.../weights/best.onnx (~5 MB)
```

### Update solution.py
```python
# In ROLL_NUMBER/solution.py
# Replace model loading:

# Before:
# self.detector = YOLO(self.model_dir / "yolo11n.pt")

# After:
self.detector = YOLO(self.model_dir / "best.pt")  # Your trained model
```

### Test Inference
```bash
python -c "
from solution import TrafficViolationDetector
detector = TrafficViolationDetector(model_dir='./models')
result = detector.predict('test_image.jpg')
print(result)
"
```

---

## Multi-Stage Training (Next Level)

Once motorcycle detection is working, train helmet & plate detectors:

```bash
# Stage 2: Helmet Detection (80 epochs, 1-1.5 hours)
python train_quick.py \
    --stage helmet \
    --data ../helmet_dataset/data.yaml \
    --epochs 80 \
    --img 416

# Stage 3: License Plate Detection (60 epochs, 45-60 min)
python train_quick.py \
    --stage plate \
    --data ../plate_dataset/data.yaml \
    --epochs 60 \
    --img 320

# OR: Full pipeline in one command
python train_quick.py --pipeline full \
    --motorcycle-data ../motorcycle_dataset/data.yaml \
    --helmet-data ../helmet_dataset/data.yaml \
    --plate-data ../plate_dataset/data.yaml
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ultralytics'"
```bash
pip install -r requirements.txt --upgrade
```

### "CUDA out of memory"
```bash
# Reduce batch size
python train_quick.py --stage motorcycle --batch 8 --data data.yaml

# Or use CPU (slower)
python train_quick.py --stage motorcycle --device cpu --data data.yaml
```

### "Low accuracy (mAP < 0.5)"
1. Check data quality: Verify 10 random image-label pairs
2. Add more data: Need ~500+ images minimum per class
3. Train longer: Increase `--epochs` to 150-200
4. Adjust batch size: Try `--batch 32` for better generalization

### "Training stuck / very slow"
```bash
# Check GPU usage
nvidia-smi

# If GPU not used, force it
python train_quick.py --stage motorcycle --device 0 --data data.yaml

# If no GPU available, CPU training takes 10-40x longer
```

---

## Next Steps

1. ✅ **First Training** (this guide)
   - [x] Install dependencies
   - [x] Prepare data
   - [x] Train motorcycle detector
   - [x] Validate

2. 🟡 **Production Training**
   - [ ] Collect real traffic violation data (~1000+ images)
   - [ ] Annotate with helmet/no-helmet labels
   - [ ] Train Stage 2 (Helmet Detection)
   - [ ] Train Stage 3 (License Plate Detection)
   - [ ] Fine-tune on edge cases

3. 🟢 **Deployment**
   - [ ] Export models to ONNX
   - [ ] Test on production hardware
   - [ ] Monitor performance
   - [ ] Iterate and improve

---

## Key Commands Reference

```bash
# Single stage training
python train_quick.py --stage motorcycle --data data.yaml

# Multi-stage training
python train_quick.py --pipeline full --motorcycle-data m.yaml --helmet-data h.yaml --plate-data p.yaml

# Validation
python train_quick.py --validate model.pt --data data.yaml

# Export model
python train_quick.py --export model.pt --format onnx

# Create sample dataset
python train_quick.py --create-sample

# Get help
python train_quick.py --help
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Motorcycle mAP50 | >85% | Primary detector |
| Helmet Accuracy | >90% | HSV fallback available |
| Plate Detection | >80% | Conservative training |
| Total Inference | <600 ms | Per image, GPU |
| Model Size | <25 MB | <250 MB project limit |

---

## Resources

- **Full Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Ultralytics Docs**: https://docs.ultralytics.com
- **Roboflow**: https://roboflow.com (free tier available)
- **CCPD Dataset**: https://github.com/detectRecog/CCPD

---

## 🎯 Summary

```
Time to first model: ~15 minutes
Time to production: ~4-6 hours (with real data)

Result: Custom traffic violation detector optimized for your environment!
```

**Ready? Let's go!** 🚀

```bash
cd ROLL_NUMBER
python train_quick.py --stage motorcycle --data ../sample_dataset/data.yaml --epochs 10
```
