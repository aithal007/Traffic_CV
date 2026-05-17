# Traffic Rule Violation Detection - AID 728

Group 23 submission for detecting two-wheeler traffic rule violations from a
single RGB street image.

## Output Format

The detector returns a JSON-compatible dictionary:

```python
{
    "violations": [
        {
            "num_riders": 1,
            "helmet_violations": 1,
            "license_plate": "KA05JW0152"
        }
    ]
}
```

Only violating two-wheelers are included in the output.

## Pipeline

1. Motorcycle/scooter detection using custom YOLO weights.
2. Global helmet/head detection using custom YOLO weights.
3. Rider assignment to each two-wheeler using trapezium ROI association.
4. Violation filtering for no-helmet and more-than-two-riders cases.
5. License plate localization using custom YOLO weights.
6. OCR using offline PaddleOCR PP-OCRv5 models.

## Final OCR Setup

The submitted solution uses PaddleOCR offline models:

- `PP-OCRv5_server_det` for text detection inside the plate crop.
- `en_PP-OCRv5_mobile_rec` for alphanumeric text recognition.

For speed, the runtime OCR path performs one PaddleOCR call on the selected
license plate crop. Document orientation, document unwarping, and textline
orientation modules are disabled because the input to OCR is already a cropped
license plate image.

## Evaluation Interface

```python
from solution import TrafficViolationDetector

model = TrafficViolationDetector(model_dir="./models")
output = model.predict(image_path)
```

All model loading is done in `__init__`. The `predict` method is stateless and
returns the required dictionary format.

## Directory Structure

```text
Group23/
  solution.py
  requirements.txt
  README.md
  models/
    last_bike_best.pt
    helmet_best_last.pt
    lp_best_last.pt
    paddleocr/
      official_models/
        PP-OCRv5_server_det/
        en_PP-OCRv5_mobile_rec/
```

## Model Size

The submitted `models/` directory is approximately 153 MB, which is below the
250 MB project limit.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

The evaluation environment must have the listed Python packages available, but
no internet access is required at runtime because all model files are packaged
inside `models/`.
