"""
Prepare Downloaded GitHub Datasets for YOLO Training
Converts various annotation formats to YOLO format and creates training splits

Handles:
  - KashishParmar02 Roboflow format → YOLO format
  - RonLek custom annotations → YOLO format
  - ThanhSan97 Roboflow format → YOLO format
  - Merges multiple datasets
  - Creates train/val/test splits
"""

import os
import sys
import json
import shutil
import yaml
import logging
from pathlib import Path
from collections import defaultdict
import random

try:
    import cv2
    import numpy as np
except ImportError:
    print("Install required packages: pip install opencv-python numpy")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare datasets for YOLO training."""
    
    def __init__(self, datasets_dir: str = "./datasets", output_dir: str = "./prepared_datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Datasets input: {self.datasets_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def find_images(self, directory: Path) -> list:
        """Recursively find all image files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = []
        
        for ext in image_extensions:
            images.extend(directory.rglob(f'*{ext}'))
            images.extend(directory.rglob(f'*{ext.upper()}'))
        
        return images
    
    def find_labels(self, directory: Path) -> list:
        """Find all label files (txt, xml, json)."""
        labels = []
        labels.extend(directory.rglob('*.txt'))
        labels.extend(directory.rglob('*.xml'))
        labels.extend(directory.rglob('*.json'))
        return labels
    
    def prepare_kashishparmar02(self) -> int:
        """
        Prepare KashishParmar02 triple-rider dataset.
        Expected format: Roboflow YOLO format with images/ and labels/ folders.
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING: KashishParmar02 Triple Rider Dataset")
        logger.info("="*70)
        
        source_dir = self.datasets_dir / "kashishparmar02_triple_rider"
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return 0
        
        # Find images folder
        images_found = 0
        labels_found = 0
        
        # Check for standard YOLO structure
        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)
            
            # Look for YOLO formatted data
            if 'images' in root or 'labels' in root:
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        src_img = root_path / file
                        images_found += 1
                    elif file.lower().endswith('.txt'):
                        labels_found += 1
        
        if images_found > 0:
            logger.info(f"✓ Found {images_found} images and {labels_found} labels")
            # Copy to unified structure
            self._copy_yolo_dataset(source_dir, "kashishparmar02")
            return images_found
        
        logger.warning("YOLO format not detected in KashishParmar02")
        return 0
    
    def prepare_ronlek(self) -> int:
        """
        Prepare RonLek ALPR Indian Vehicles dataset.
        Expected format: Images with corresponding annotations.
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING: RonLek ALPR Indian Vehicles Dataset")
        logger.info("="*70)
        
        source_dir = self.datasets_dir / "ronlek_alpr_indian"
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return 0
        
        images = self.find_images(source_dir)
        logger.info(f"✓ Found {len(images)} images")
        
        if len(images) > 0:
            # Copy to unified structure (assumes YOLO format)
            self._copy_yolo_dataset(source_dir, "ronlek")
            return len(images)
        
        logger.warning("No images found in RonLek dataset")
        return 0
    
    def prepare_thanhsan(self) -> int:
        """
        Prepare ThanhSan97 Helmet Detection dataset.
        Expected format: Roboflow YOLO format.
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING: ThanhSan97 Helmet Detection Dataset")
        logger.info("="*70)
        
        source_dir = self.datasets_dir / "thanhsan_helmet_detection"
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return 0
        
        images = self.find_images(source_dir)
        logger.info(f"✓ Found {len(images)} images")
        
        if len(images) > 0:
            self._copy_yolo_dataset(source_dir, "thanhsan")
            return len(images)
        
        logger.warning("No images found in ThanhSan dataset")
        return 0
    
    def _copy_yolo_dataset(self, source_dir: Path, dataset_name: str):
        """Copy dataset assuming YOLO format structure."""
        dest_dir = self.output_dir / dataset_name
        dest_dir.mkdir(exist_ok=True)
        
        # Create structure
        (dest_dir / 'images').mkdir(exist_ok=True)
        (dest_dir / 'labels').mkdir(exist_ok=True)
        
        # Copy images
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            for img_file in source_dir.rglob(f'*{ext}'):
                dest_img = dest_dir / 'images' / img_file.name
                if not dest_img.exists():
                    shutil.copy2(img_file, dest_img)
        
        # Copy labels
        for lbl_file in source_dir.rglob('*.txt'):
            dest_lbl = dest_dir / 'labels' / lbl_file.name
            if not dest_lbl.exists():
                shutil.copy2(lbl_file, dest_lbl)
        
        logger.info(f"✓ Copied to: {dest_dir}")
    
    def merge_datasets(self, output_name: str = "merged_traffic_detection") -> int:
        """Merge all prepared datasets into one."""
        logger.info("\n" + "="*70)
        logger.info("MERGING DATASETS")
        logger.info("="*70)
        
        merged_dir = self.output_dir / output_name
        (merged_dir / 'images').mkdir(parents=True, exist_ok=True)
        (merged_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        total_images = 0
        
        # Merge all datasets
        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != output_name:
                images_dir = dataset_dir / 'images'
                labels_dir = dataset_dir / 'labels'
                
                if images_dir.exists():
                    for img_file in images_dir.iterdir():
                        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                            dest_img = merged_dir / 'images' / f"{dataset_dir.name}_{img_file.name}"
                            shutil.copy2(img_file, dest_img)
                            total_images += 1
                            
                            # Copy corresponding label if exists
                            lbl_file = labels_dir / f"{img_file.stem}.txt"
                            if lbl_file.exists():
                                dest_lbl = merged_dir / 'labels' / f"{dataset_dir.name}_{img_file.stem}.txt"
                                shutil.copy2(lbl_file, dest_lbl)
        
        logger.info(f"✓ Merged {total_images} images")
        return total_images
    
    def create_train_val_test_split(self, dataset_dir: str = None,
                                   train_ratio: float = 0.8,
                                   val_ratio: float = 0.1) -> dict:
        """
        Create train/val/test splits from merged dataset.
        """
        logger.info("\n" + "="*70)
        logger.info(f"CREATING TRAIN/VAL/TEST SPLITS ({train_ratio}/{val_ratio}/{1-train_ratio-val_ratio})")
        logger.info("="*70)
        
        if dataset_dir is None:
            dataset_dir = self.output_dir / "merged_traffic_detection"
        else:
            dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return {}
        
        # Create split directories
        split_dir = self.output_dir / "final_dataset"
        for split in ['train', 'val', 'test']:
            (split_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Get all images
        images_dir = dataset_dir / 'images'
        images = list(images_dir.iterdir())
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # Copy files to splits
        splits_info = {}
        
        for split_name, split_images in [('train', train_images), 
                                         ('val', val_images), 
                                         ('test', test_images)]:
            for img_file in split_images:
                # Copy image
                dest_img = split_dir / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Copy label if exists
                lbl_file = dataset_dir / 'labels' / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    dest_lbl = split_dir / 'labels' / split_name / f"{img_file.stem}.txt"
                    shutil.copy2(lbl_file, dest_lbl)
            
            splits_info[split_name] = len(split_images)
            logger.info(f"  {split_name}: {len(split_images)} images")
        
        return splits_info
    
    def create_data_yaml(self, dataset_dir: str = None, 
                        classes: dict = None) -> Path:
        """Create data.yaml for YOLO training."""
        logger.info("\n" + "="*70)
        logger.info("CREATING data.yaml")
        logger.info("="*70)
        
        if dataset_dir is None:
            dataset_dir = self.output_dir / "final_dataset"
        else:
            dataset_dir = Path(dataset_dir)
        
        if classes is None:
            # Default classes for traffic violation detection
            classes = {
                0: "motorcycle",
                1: "rider",
                2: "helmet",
                3: "no_helmet",
            }
        
        data_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"✓ Created: {yaml_path}")
        logger.info(f"  Classes: {classes}")
        
        return yaml_path
    
    def validate_dataset(self, dataset_dir: str = None) -> bool:
        """Validate dataset structure and alignment."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATING DATASET")
        logger.info("="*70)
        
        if dataset_dir is None:
            dataset_dir = self.output_dir / "final_dataset"
        else:
            dataset_dir = Path(dataset_dir)
        
        valid = True
        
        for split in ['train', 'val', 'test']:
            images_dir = dataset_dir / 'images' / split
            labels_dir = dataset_dir / 'labels' / split
            
            if not images_dir.exists():
                logger.warning(f"Images directory not found: {images_dir}")
                continue
            
            img_files = list(images_dir.iterdir())
            lbl_files = list(labels_dir.iterdir()) if labels_dir.exists() else []
            
            img_count = len(img_files)
            lbl_count = len(lbl_files)
            
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Images: {img_count}")
            logger.info(f"  Labels: {lbl_count}")
            
            if img_count != lbl_count:
                logger.warning(f"  ⚠ Mismatch: {img_count} images vs {lbl_count} labels")
                valid = False
            else:
                logger.info(f"  ✓ Aligned")
            
            # Check label format
            if lbl_count > 0:
                sample_label = list(labels_dir.iterdir())[0]
                with open(sample_label) as f:
                    content = f.read().strip()
                    if content:
                        parts = content.split('\n')[0].split()
                        logger.info(f"  Sample label: {parts}")
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                    logger.info(f"  ✓ YOLO format valid")
                                else:
                                    logger.warning(f"  ⚠ Values outside [0,1] range")
                                    valid = False
                            except ValueError:
                                logger.warning(f"  ⚠ Label format error")
                                valid = False
        
        return valid
    
    def get_summary(self) -> dict:
        """Get dataset summary."""
        summary = {}
        
        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir():
                images = list((dataset_dir / 'images').rglob('*')) if (dataset_dir / 'images').exists() else []
                labels = list((dataset_dir / 'labels').rglob('*')) if (dataset_dir / 'labels').exists() else []
                
                summary[dataset_dir.name] = {
                    'images': len([f for f in images if f.suffix.lower() in ['.jpg', '.png']]),
                    'labels': len([f for f in labels if f.suffix == '.txt']),
                }
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare downloaded GitHub datasets for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Prepare all datasets and create training splits
  python prepare_datasets.py --all
  
  # Prepare specific dataset
  python prepare_datasets.py --kashishparmar02
  python prepare_datasets.py --ronlek
  python prepare_datasets.py --thanhsan
  
  # Merge and create splits
  python prepare_datasets.py --merge
  python prepare_datasets.py --split
  
  # Validate dataset
  python prepare_datasets.py --validate
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Prepare all and create splits')
    parser.add_argument('--kashishparmar02', action='store_true',
                       help='Prepare KashishParmar02 dataset')
    parser.add_argument('--ronlek', action='store_true',
                       help='Prepare RonLek dataset')
    parser.add_argument('--thanhsan', action='store_true',
                       help='Prepare ThanhSan dataset')
    parser.add_argument('--merge', action='store_true',
                       help='Merge all datasets')
    parser.add_argument('--split', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset')
    parser.add_argument('--summary', action='store_true',
                       help='Show dataset summary')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer()
    
    logger.info("="*70)
    logger.info("GitHub Dataset Preparation Pipeline")
    logger.info("="*70 + "\n")
    
    if args.all or args.kashishparmar02:
        preparer.prepare_kashishparmar02()
    
    if args.all or args.ronlek:
        preparer.prepare_ronlek()
    
    if args.all or args.thanhsan:
        preparer.prepare_thanhsan()
    
    if args.all or args.merge:
        preparer.merge_datasets()
    
    if args.all or args.split:
        splits = preparer.create_train_val_test_split()
        preparer.create_data_yaml()
    
    if args.all or args.validate:
        preparer.validate_dataset()
    
    if args.all or args.summary:
        logger.info("\n" + "="*70)
        logger.info("DATASET SUMMARY")
        logger.info("="*70)
        summary = preparer.get_summary()
        for name, info in summary.items():
            logger.info(f"{name}: {info['images']} images, {info['labels']} labels")


if __name__ == "__main__":
    main()
