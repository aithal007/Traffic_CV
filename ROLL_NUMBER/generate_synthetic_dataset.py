"""
Generate Synthetic Annotated Dataset for Testing
Creates properly formatted YOLO annotations for training pipeline demonstration
"""

import os
import random
from pathlib import Path
import yaml

def create_synthetic_dataset(num_train=100, num_val=20, num_test=20):
    """
    Create synthetic YOLO dataset for training pipeline testing.
    Generates random images with valid YOLO format annotations.
    """
    
    try:
        import numpy as np
        import cv2
    except ImportError:
        print("pip install numpy opencv-python")
        return False
    
    dataset_dir = Path("synthetic_dataset")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Class definitions
    classes = {0: 'motorcycle', 1: 'rider', 2: 'helmet', 3: 'no_helmet'}
    
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DATASET FOR TRAINING")
    print("="*70)
    
    def generate_image_with_annotations(output_dir, split_name, num_images):
        """Generate synthetic images with YOLO annotations."""
        
        for i in range(num_images):
            # Create random image (640x640)
            img = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            # Add some structure (simulate traffic scene)
            cv2.rectangle(img, (50, 50), (600, 400), (100, 150, 100), 5)  # Road
            cv2.rectangle(img, (100, 150), (300, 350), (50, 100, 200), -1)  # Motorcycle
            cv2.circle(img, (180, 300), 40, (150, 150, 150), -1)  # Wheel
            cv2.circle(img, (250, 280), 30, (200, 100, 50), -1)  # Rider
            
            # Random annotations (YOLO format: class_id x_center y_center width height)
            annotations = []
            
            # Add motorcycle (required)
            moto_x, moto_y = random.uniform(0.1, 0.5), random.uniform(0.2, 0.6)
            moto_w, moto_h = random.uniform(0.2, 0.4), random.uniform(0.25, 0.45)
            annotations.append(f"0 {moto_x:.4f} {moto_y:.4f} {moto_w:.4f} {moto_h:.4f}")
            
            # Add 1-3 riders
            num_riders = random.randint(1, 3)
            for _ in range(num_riders):
                rider_x = moto_x + random.uniform(-0.15, 0.15)
                rider_y = moto_y + random.uniform(-0.2, 0.2)
                rider_w, rider_h = random.uniform(0.08, 0.15), random.uniform(0.15, 0.3)
                
                # Clamp to valid range
                rider_x = max(0.05, min(0.95, rider_x))
                rider_y = max(0.1, min(0.9, rider_y))
                
                annotations.append(f"1 {rider_x:.4f} {rider_y:.4f} {rider_w:.4f} {rider_h:.4f}")
                
                # Random helmet presence
                if random.random() > 0.3:
                    helmet_x = rider_x
                    helmet_y = rider_y - 0.1
                    helmet_w, helmet_h = random.uniform(0.05, 0.12), random.uniform(0.08, 0.15)
                    helmet_x = max(0.05, min(0.95, helmet_x))
                    helmet_y = max(0.05, min(0.95, helmet_y))
                    annotations.append(f"2 {helmet_x:.4f} {helmet_y:.4f} {helmet_w:.4f} {helmet_h:.4f}")
            
            # Save image
            img_path = output_dir / 'images' / split_name / f"image_{i:05d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Save annotations
            lbl_path = output_dir / 'labels' / split_name / f"image_{i:05d}.txt"
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(annotations))
        
        print(f"✓ Generated {num_images} {split_name} images with annotations")
    
    # Generate splits
    generate_image_with_annotations(dataset_dir, 'train', num_train)
    generate_image_with_annotations(dataset_dir, 'val', num_val)
    generate_image_with_annotations(dataset_dir, 'test', num_test)
    
    # Create data.yaml
    data_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"\n✓ Created data.yaml with {len(classes)} classes")
    print(f"✓ Dataset ready at: {dataset_dir}")
    print(f"\nTotal images: {num_train + num_val + num_test}")
    print(f"Train: {num_train}, Val: {num_val}, Test: {num_test}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for training")
    parser.add_argument('--train', type=int, default=100, help="Training images")
    parser.add_argument('--val', type=int, default=20, help="Validation images")
    parser.add_argument('--test', type=int, default=20, help="Test images")
    
    args = parser.parse_args()
    
    create_synthetic_dataset(args.train, args.val, args.test)
