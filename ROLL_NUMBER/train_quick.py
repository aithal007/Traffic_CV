"""
Quick Training Starter Script
Provides CLI interface for easy model training

Usage:
    python train_quick.py --stage motorcycle --data path/to/data.yaml
    python train_quick.py --pipeline full --motorcycle-data m.yaml --helmet-data h.yaml
"""

import argparse
import sys
from pathlib import Path
import logging

from train import TrainingPipeline, create_sample_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Train motorcycle detector (Stage 1)
  python train_quick.py --stage motorcycle --data ./dataset/data.yaml
  
  # Train helmet detector (Stage 2)
  python train_quick.py --stage helmet --data ./helmet_dataset/data.yaml --epochs 80
  
  # Train plate detector (Stage 3)
  python train_quick.py --stage plate --data ./plate_dataset/data.yaml --epochs 60
  
  # Train all stages (full pipeline)
  python train_quick.py --pipeline full \\
    --motorcycle-data ./motorcycle_dataset/data.yaml \\
    --helmet-data ./helmet_dataset/data.yaml \\
    --plate-data ./plate_dataset/data.yaml
  
  # Validate trained model
  python train_quick.py --validate ./training_output/.../best.pt \\
    --data ./dataset/data.yaml
  
  # Export model to ONNX
  python train_quick.py --export ./training_output/.../best.pt --format onnx

Tips:
  - Use GPU: CUDA automatically detected
  - Monitor training: Check training_output/ directory
  - Early stopping: Press Ctrl+C to stop and save
  - For multi-GPU: Use --device 0,1,2,3 (comma-separated)
        """
    )
    
    # Main pipeline options
    pipeline_group = parser.add_argument_group('Pipeline Mode')
    pipeline_group.add_argument('--stage', 
                               choices=['motorcycle', 'helmet', 'plate'],
                               help='Train single stage')
    pipeline_group.add_argument('--pipeline',
                               choices=['full', 'multi-stage'],
                               help='Run full multi-stage training')
    pipeline_group.add_argument('--validate',
                               type=str,
                               help='Validate model (provide model path)')
    pipeline_group.add_argument('--export',
                               type=str,
                               help='Export model (provide model path)')
    
    # Dataset paths
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--data',
                           type=str,
                           help='Path to data.yaml (single stage)')
    data_group.add_argument('--motorcycle-data',
                           type=str,
                           help='Path to motorcycle detection data.yaml')
    data_group.add_argument('--helmet-data',
                           type=str,
                           help='Path to helmet detection data.yaml')
    data_group.add_argument('--plate-data',
                           type=str,
                           help='Path to license plate detection data.yaml')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs',
                            type=int,
                            default=100,
                            help='Number of epochs (default: 100)')
    train_group.add_argument('--batch',
                            type=int,
                            default=16,
                            help='Batch size (default: 16)')
    train_group.add_argument('--img',
                            type=int,
                            default=640,
                            help='Input image size (default: 640)')
    train_group.add_argument('--device',
                            type=str,
                            default='0',
                            help='GPU device ID (default: 0, use "cpu" for CPU)')
    train_group.add_argument('--patience',
                            type=int,
                            default=20,
                            help='Early stopping patience (default: 20)')
    
    # Export options
    export_group = parser.add_argument_group('Export Configuration')
    export_group.add_argument('--format',
                             choices=['pt', 'onnx', 'torchscript', 'tflite', 'pb'],
                             default='onnx',
                             help='Export format (default: onnx)')
    
    # Misc options
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--project-dir',
                           type=str,
                           default='./training_output',
                           help='Output directory for training (default: ./training_output)')
    misc_group.add_argument('--base-model',
                           type=str,
                           default='yolo11n.pt',
                           help='Base model name (default: yolo11n.pt)')
    misc_group.add_argument('--create-sample',
                           action='store_true',
                           help='Create sample dataset structure')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        base_model=args.base_model,
        project_dir=args.project_dir
    )
    
    # Handle sample dataset creation
    if args.create_sample:
        logger.info("Creating sample dataset structure...")
        create_sample_dataset()
        logger.info("Sample dataset created. Add your data and labels.")
        return
    
    # Single stage training
    if args.stage:
        if not args.data:
            logger.error("--data required for single stage training")
            sys.exit(1)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training: {args.stage.upper()} Detection")
        logger.info(f"{'='*70}")
        logger.info(f"Data: {args.data}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch Size: {args.batch}")
        logger.info(f"Image Size: {args.img}")
        logger.info(f"Device: {args.device}")
        logger.info(f"{'='*70}\n")
        
        if args.stage == 'motorcycle':
            best_model = pipeline.train_motorcycle_detector(
                args.data,
                epochs=args.epochs,
                batch_size=args.batch,
                img_size=args.img,
                patience=args.patience
            )
        elif args.stage == 'helmet':
            best_model = pipeline.train_helmet_detector(
                args.data,
                epochs=args.epochs,
                batch_size=args.batch,
                img_size=args.img,
                patience=args.patience
            )
        elif args.stage == 'plate':
            best_model = pipeline.train_plate_detector(
                args.data,
                epochs=args.epochs,
                batch_size=args.batch,
                img_size=args.img,
                patience=args.patience
            )
        
        if best_model:
            logger.info(f"\n✓ Training completed successfully!")
            logger.info(f"Best model: {best_model}")
        else:
            logger.error(f"\n✗ Training failed!")
            sys.exit(1)
    
    # Multi-stage training
    elif args.pipeline:
        if not (args.motorcycle_data and args.helmet_data and args.plate_data):
            logger.error("--motorcycle-data, --helmet-data, and --plate-data required for full pipeline")
            sys.exit(1)
        
        logger.info(f"\n{'='*70}")
        logger.info("MULTI-STAGE TRAINING PIPELINE")
        logger.info(f"{'='*70}")
        logger.info("Stage 1: Motorcycle Detection")
        logger.info("Stage 2: Helmet Detection")
        logger.info("Stage 3: License Plate Detection")
        logger.info(f"{'='*70}\n")
        
        pipeline.run_full_pipeline(
            motorcycle_data_yaml=args.motorcycle_data,
            helmet_data_yaml=args.helmet_data,
            plate_data_yaml=args.plate_data,
            epochs_per_stage=(args.epochs, int(args.epochs * 0.8), int(args.epochs * 0.6))
        )
    
    # Validation
    elif args.validate:
        if not args.data:
            logger.error("--data required for validation")
            sys.exit(1)
        
        logger.info(f"\n{'='*70}")
        logger.info("MODEL VALIDATION")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {args.validate}")
        logger.info(f"Data: {args.data}")
        logger.info(f"{'='*70}\n")
        
        metrics = pipeline.validate_model(
            args.validate,
            args.data,
            img_size=args.img
        )
        
        logger.info("\n✓ Validation completed!")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value}")
    
    # Export
    elif args.export:
        logger.info(f"\n{'='*70}")
        logger.info("MODEL EXPORT")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {args.export}")
        logger.info(f"Format: {args.format}")
        logger.info(f"{'='*70}\n")
        
        export_path = pipeline.export_model(args.export, format=args.format)
        
        if export_path:
            logger.info(f"\n✓ Export completed successfully!")
            logger.info(f"Exported model: {export_path}")
        else:
            logger.error(f"\n✗ Export failed!")
            sys.exit(1)
    
    else:
        parser.print_help()
        logger.info("\nNo action specified. Use --help for examples.")


if __name__ == "__main__":
    main()
