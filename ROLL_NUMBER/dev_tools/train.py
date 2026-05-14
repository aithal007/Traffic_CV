"""
Advanced Training Pipeline for Traffic Violation Detection System
Supports fine-tuning on custom datasets for improved robustness.

Features:
  - YOLO11n fine-tuning on motorcycle/rider/helmet detection
  - Multi-stage training (detection → helmet → plate)
  - Automatic dataset preparation and validation
  - Comprehensive logging and metrics
  - Model checkpointing and validation
"""

import os
import sys
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from ultralytics import YOLO
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Comprehensive training pipeline for traffic violation detection.
    Supports multi-stage training and dataset validation.
    """

    def __init__(self, base_model: str = "yolo11n.pt", project_dir: str = "./training_output"):
        """
        Initialize training pipeline.
        
        Args:
            base_model: Base YOLO model name (yolo11n, yolo11s, yolo11m)
            project_dir: Output directory for training runs
        """
        self.base_model = base_model
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.datasets_dir = self.project_dir / "datasets"
        self.models_dir = self.project_dir / "models"
        self.logs_dir = self.project_dir / "logs"
        self.checkpoints_dir = self.project_dir / "checkpoints"
        
        for d in [self.datasets_dir, self.models_dir, self.logs_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Training Pipeline initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Project directory: {self.project_dir}")

    def prepare_dataset(self, data_yaml: str, dataset_name: str = "motorcycle_detection") -> bool:
        """
        Prepare and validate dataset in YOLO format.
        
        Expected structure:
        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
        
        Args:
            data_yaml: Path to data.yaml configuration file
            dataset_name: Name of the dataset
            
        Returns:
            True if dataset is valid, False otherwise
        """
        logger.info(f"Preparing dataset: {dataset_name}")
        
        data_yaml_path = Path(data_yaml)
        if not data_yaml_path.exists():
            logger.error(f"Data YAML not found: {data_yaml}")
            return False
        
        # Load YAML configuration
        with open(data_yaml_path) as f:
            data_config = yaml.safe_load(f)
        
        logger.info(f"Dataset configuration:")
        logger.info(f"  Classes: {data_config.get('nc')}")
        logger.info(f"  Names: {data_config.get('names')}")
        
        # Validate dataset structure
        dataset_root = data_yaml_path.parent
        required_splits = ['train', 'val']
        
        for split in required_splits:
            images_dir = dataset_root / 'images' / split
            labels_dir = dataset_root / 'labels' / split
            
            if not images_dir.exists():
                logger.warning(f"Missing {split} images directory: {images_dir}")
                continue
            
            img_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
            label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            logger.info(f"  {split}: {img_count} images, {label_count} labels")
            
            if img_count != label_count:
                logger.warning(f"Mismatch in {split} split: {img_count} images vs {label_count} labels")
        
        # Copy dataset to training directory
        dest_path = self.datasets_dir / dataset_name
        if dest_path.exists():
            logger.info(f"Dataset already exists at {dest_path}")
        else:
            logger.info(f"Copying dataset to {dest_path}")
            shutil.copytree(dataset_root, dest_path)
        
        return True

    def create_dataset_config(self, 
                             data_dir: str,
                             classes: Dict[int, str],
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1) -> Path:
        """
        Create YOLO dataset configuration YAML file.
        
        Args:
            data_dir: Path to dataset directory
            classes: Dictionary mapping class indices to names
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            
        Returns:
            Path to created data.yaml file
        """
        data_dir = Path(data_dir)
        
        # Calculate test ratio
        test_ratio = 1.0 - train_ratio - val_ratio
        
        config = {
            'path': str(data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(classes),
            'names': classes
        }
        
        config_path = data_dir / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created dataset configuration: {config_path}")
        return config_path

    def train_motorcycle_detector(self,
                                 data_yaml: str,
                                 epochs: int = 100,
                                 batch_size: int = 16,
                                 img_size: int = 640,
                                 patience: int = 20) -> Optional[str]:
        """
        Train motorcycle detection model (Stage 1).
        
        Detects: motorcycles, riders
        
        Args:
            data_yaml: Path to data.yaml for motorcycle detection
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            patience: Early stopping patience
            
        Returns:
            Path to best trained model
        """
        logger.info("=" * 70)
        logger.info("STAGE 1: MOTORCYCLE DETECTION TRAINING")
        logger.info("=" * 70)
        
        try:
            # Load base model
            logger.info(f"Loading base model: {self.base_model}")
            model = YOLO(self.base_model)
            
            # Training configuration
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                device=self.device,
                project=str(self.project_dir / "motorcycle_detector"),
                name=f"run_{self.timestamp}",
                
                # Learning rate and optimization
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                
                # Augmentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10,
                translate=0.1,
                scale=0.5,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0,
                
                # Validation and checkpointing
                val=True,
                save=True,
                save_period=10,
                verbose=True,
                
                # Performance
                workers=4,
                cache=True,
            )
            
            # Get best model path
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            logger.info(f"Motorcycle detector training completed!")
            logger.info(f"Best model: {best_model}")
            
            return str(best_model)
            
        except Exception as e:
            logger.error(f"Motorcycle detector training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_helmet_detector(self,
                            data_yaml: str,
                            epochs: int = 80,
                            batch_size: int = 32,
                            img_size: int = 416,
                            patience: int = 15) -> Optional[str]:
        """
        Train helmet detection model (Stage 2).
        
        Detects: helmets, no-helmet (for refinement)
        
        Args:
            data_yaml: Path to data.yaml for helmet detection
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size (smaller for faster helmet detection)
            patience: Early stopping patience
            
        Returns:
            Path to best trained model
        """
        logger.info("=" * 70)
        logger.info("STAGE 2: HELMET DETECTION TRAINING")
        logger.info("=" * 70)
        
        try:
            # Load base model
            logger.info(f"Loading base model: {self.base_model}")
            model = YOLO(self.base_model)
            
            # Training configuration
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                device=self.device,
                project=str(self.project_dir / "helmet_detector"),
                name=f"run_{self.timestamp}",
                
                # Learning rate
                lr0=0.01,
                lrf=0.01,
                
                # Augmentation (more aggressive for helmet variety)
                hsv_h=0.02,
                hsv_s=0.8,
                hsv_v=0.5,
                degrees=15,
                translate=0.15,
                scale=0.6,
                flipud=0.5,
                fliplr=0.5,
                
                # Validation
                val=True,
                save=True,
                save_period=5,
                verbose=True,
                
                workers=4,
                cache=True,
            )
            
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            logger.info(f"Helmet detector training completed!")
            logger.info(f"Best model: {best_model}")
            
            return str(best_model)
            
        except Exception as e:
            logger.error(f"Helmet detector training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_plate_detector(self,
                            data_yaml: str,
                            epochs: int = 60,
                            batch_size: int = 32,
                            img_size: int = 320,
                            patience: int = 12) -> Optional[str]:
        """
        Train license plate detection model (Stage 3).
        
        Detects: license plates
        
        Args:
            data_yaml: Path to data.yaml for plate detection
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size (smaller for plate detection)
            patience: Early stopping patience
            
        Returns:
            Path to best trained model
        """
        logger.info("=" * 70)
        logger.info("STAGE 3: LICENSE PLATE DETECTION TRAINING")
        logger.info("=" * 70)
        
        try:
            # Load base model
            logger.info(f"Loading base model: {self.base_model}")
            model = YOLO(self.base_model)
            
            # Training configuration
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                device=self.device,
                project=str(self.project_dir / "plate_detector"),
                name=f"run_{self.timestamp}",
                
                # Conservative augmentation for plate (need to preserve text)
                hsv_h=0.01,
                hsv_s=0.5,
                hsv_v=0.3,
                degrees=5,
                translate=0.05,
                scale=0.3,
                flipud=0.0,  # No vertical flip (plate orientation matters)
                fliplr=0.5,
                
                lr0=0.01,
                lrf=0.01,
                
                val=True,
                save=True,
                save_period=5,
                verbose=True,
                
                workers=4,
                cache=True,
            )
            
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            logger.info(f"Plate detector training completed!")
            logger.info(f"Best model: {best_model}")
            
            return str(best_model)
            
        except Exception as e:
            logger.error(f"Plate detector training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_model(self, model_path: str, data_yaml: str, img_size: int = 640) -> Dict:
        """
        Validate trained model on validation/test set.
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to data.yaml
            img_size: Input image size
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Validating model: {model_path}")
        
        try:
            model = YOLO(model_path)
            results = model.val(data=data_yaml, imgsz=img_size, device=self.device)
            
            metrics = {
                'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else None,
                'mAP50_95': float(results.box.map) if hasattr(results.box, 'map') else None,
                'precision': float(results.box.p.mean()) if hasattr(results.box, 'p') else None,
                'recall': float(results.box.r.mean()) if hasattr(results.box, 'r') else None,
            }
            
            logger.info(f"Validation Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def export_model(self, model_path: str, format: str = "onnx") -> Optional[str]:
        """
        Export trained model to different formats.
        
        Supported formats: pt, onnx, torchscript, tflite, pb, etc.
        
        Args:
            model_path: Path to trained model
            format: Export format (default: ONNX for deployment)
            
        Returns:
            Path to exported model
        """
        logger.info(f"Exporting model to {format}: {model_path}")
        
        try:
            model = YOLO(model_path)
            export_path = model.export(format=format)
            logger.info(f"Model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_full_pipeline(self,
                         motorcycle_data_yaml: Optional[str] = None,
                         helmet_data_yaml: Optional[str] = None,
                         plate_data_yaml: Optional[str] = None,
                         epochs_per_stage: Tuple[int, int, int] = (100, 80, 60)):
        """
        Run complete multi-stage training pipeline.
        
        Args:
            motorcycle_data_yaml: Path to motorcycle detection dataset
            helmet_data_yaml: Path to helmet detection dataset
            plate_data_yaml: Path to plate detection dataset
            epochs_per_stage: Epochs for (motorcycle, helmet, plate) stages
        """
        logger.info("=" * 70)
        logger.info("STARTING MULTI-STAGE TRAINING PIPELINE")
        logger.info("=" * 70)
        
        results = {
            'timestamp': self.timestamp,
            'stages': {}
        }
        
        # Stage 1: Motorcycle Detection
        if motorcycle_data_yaml:
            logger.info("\n[STAGE 1/3] Training Motorcycle Detector...")
            best_moto = self.train_motorcycle_detector(
                motorcycle_data_yaml,
                epochs=epochs_per_stage[0]
            )
            if best_moto:
                results['stages']['motorcycle'] = {
                    'model': best_moto,
                    'status': 'completed'
                }
                # Validate
                metrics = self.validate_model(best_moto, motorcycle_data_yaml)
                results['stages']['motorcycle']['metrics'] = metrics
            else:
                results['stages']['motorcycle'] = {'status': 'failed'}
        
        # Stage 2: Helmet Detection
        if helmet_data_yaml:
            logger.info("\n[STAGE 2/3] Training Helmet Detector...")
            best_helmet = self.train_helmet_detector(
                helmet_data_yaml,
                epochs=epochs_per_stage[1]
            )
            if best_helmet:
                results['stages']['helmet'] = {
                    'model': best_helmet,
                    'status': 'completed'
                }
                # Validate
                metrics = self.validate_model(best_helmet, helmet_data_yaml)
                results['stages']['helmet']['metrics'] = metrics
            else:
                results['stages']['helmet'] = {'status': 'failed'}
        
        # Stage 3: License Plate Detection
        if plate_data_yaml:
            logger.info("\n[STAGE 3/3] Training License Plate Detector...")
            best_plate = self.train_plate_detector(
                plate_data_yaml,
                epochs=epochs_per_stage[2]
            )
            if best_plate:
                results['stages']['plate'] = {
                    'model': best_plate,
                    'status': 'completed'
                }
                # Validate
                metrics = self.validate_model(best_plate, plate_data_yaml)
                results['stages']['plate']['metrics'] = metrics
            else:
                results['stages']['plate'] = {'status': 'failed'}
        
        # Save training summary
        summary_path = self.logs_dir / f"training_summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Training summary saved to: {summary_path}")
        
        return results


def create_sample_dataset():
    """
    Create sample dataset structure for testing.
    In production, you would populate this with real annotated data.
    """
    logger.info("Creating sample dataset structure...")
    
    dataset_dir = Path("./sample_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Create sample data.yaml
    data_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': {0: 'motorcycle', 1: 'rider'}
    }
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    logger.info(f"Sample dataset structure created at: {dataset_dir}")
    logger.info("Note: Add your annotated images and labels to this directory")
    
    return dataset_dir


if __name__ == "__main__":
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        base_model="yolo11n.pt",
        project_dir="./training_output"
    )
    
    logger.info("=" * 70)
    logger.info("TRAFFIC VIOLATION DETECTION - TRAINING SYSTEM")
    logger.info("=" * 70)
    logger.info("\nUsage Examples:\n")
    
    logger.info("1. SINGLE STAGE TRAINING (Motorcycle Detection):")
    logger.info("   python train.py --stage motorcycle --data path/to/data.yaml\n")
    
    logger.info("2. MULTI-STAGE TRAINING (All stages):")
    logger.info("   python train.py --pipeline full \\")
    logger.info("     --motorcycle-data path/to/moto/data.yaml \\")
    logger.info("     --helmet-data path/to/helmet/data.yaml \\")
    logger.info("     --plate-data path/to/plate/data.yaml\n")
    
    logger.info("3. VALIDATION:")
    logger.info("   python train.py --validate path/to/model.pt --data path/to/data.yaml\n")
    
    logger.info("4. EXPORT MODEL:")
    logger.info("   python train.py --export path/to/model.pt --format onnx\n")
    
    # Create sample dataset structure
    create_sample_dataset()
    
    logger.info("\n" + "=" * 70)
    logger.info("Setup complete! Next steps:")
    logger.info("1. Prepare your dataset in YOLO format")
    logger.info("2. Place it in the appropriate directory")
    logger.info("3. Create data.yaml configuration")
    logger.info("4. Run training pipeline")
    logger.info("=" * 70)
