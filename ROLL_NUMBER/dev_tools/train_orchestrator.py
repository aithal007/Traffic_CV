#!/usr/bin/env python3
"""
Master Training Orchestrator
One command to download, prepare, and train models using GitHub datasets
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run shell command with error handling."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        logger.info(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error code {e.returncode}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master Training Orchestrator - Download, Prepare, Train All-in-One",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START (3 commands):

  # 1. Full pipeline (download + prepare + train)
  python train_orchestrator.py --full-pipeline

  # 2. Or step-by-step
  python train_orchestrator.py --download
  python train_orchestrator.py --prepare
  python train_orchestrator.py --train --stage motorcycle

  # 3. Or deploy trained models
  python train_orchestrator.py --validate best.pt
  python train_orchestrator.py --export best.pt

EXAMPLES:

  # Complete training from scratch (2-3 hours)
  python train_orchestrator.py --full-pipeline

  # Only download and prepare (no training yet)
  python train_orchestrator.py --download --prepare

  # Train after manual dataset preparation
  python train_orchestrator.py --train --stage motorcycle --epochs 100

  # Monitor running training
  python train_orchestrator.py --tensorboard

  # Quick test (sample data)
  python train_orchestrator.py --quick-test
        """
    )
    
    # Main workflow
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Complete workflow: download → prepare → train')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with sample data (5-10 min)')
    
    # Individual steps
    parser.add_argument('--download', action='store_true',
                       help='Download GitHub datasets only')
    parser.add_argument('--prepare', action='store_true',
                       help='Prepare datasets to YOLO format')
    parser.add_argument('--train', action='store_true',
                       help='Start training')
    parser.add_argument('--validate', type=str,
                       help='Validate trained model')
    parser.add_argument('--export', type=str,
                       help='Export model to ONNX')
    
    # Training options
    parser.add_argument('--stage', choices=['motorcycle', 'helmet', 'plate'],
                       default='motorcycle',
                       help='Training stage')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU device (0, 1, 2... or cpu)')
    
    # Utilities
    parser.add_argument('--tensorboard', action='store_true',
                       help='Start TensorBoard monitoring')
    parser.add_argument('--summary', action='store_true',
                       help='Show dataset summary')
    parser.add_argument('--clean', action='store_true',
                       help='Clean output directories')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("TRAFFIC VIOLATION DETECTION - MASTER TRAINING ORCHESTRATOR")
    logger.info("="*70)
    
    # Ensure we're in ROLL_NUMBER directory
    if not Path("train.py").exists():
        logger.error("Must run from ROLL_NUMBER directory (contains train.py)")
        sys.exit(1)
    
    # Quick test workflow
    if args.quick_test:
        logger.info("Starting QUICK TEST workflow (5-10 minutes with GPU)")
        
        # Create sample dataset
        if not Path("sample_dataset").exists():
            if not run_command("python train_quick.py --create-sample", 
                             "Create sample dataset"):
                sys.exit(1)
        
        # Train on sample
        if not run_command(
            f"python train_quick.py --stage {args.stage} "
            f"--data ../sample_dataset/data.yaml "
            f"--epochs 10 --batch 8 --device {args.device}",
            f"Train {args.stage} detector on sample data"):
            sys.exit(1)
        
        logger.info("\n✓ Quick test completed!")
        return
    
    # Full pipeline workflow
    if args.full_pipeline:
        logger.info("Starting FULL PIPELINE (download → prepare → train)")
        logger.info("Estimated time: 2-3 hours\n")
        
        # Step 1: Download
        if not run_command("python download_github_datasets.py --all",
                         "Download GitHub datasets"):
            logger.warning("Dataset download had issues, but continuing...")
        
        # Step 2: Prepare
        if not run_command("python prepare_datasets.py --all",
                         "Prepare datasets to YOLO format"):
            logger.error("Dataset preparation failed")
            sys.exit(1)
        
        # Step 3: Train
        if not run_command(
            f"python train_quick.py --stage {args.stage} "
            f"--data prepared_datasets/final_dataset/data.yaml "
            f"--epochs {args.epochs} --batch {args.batch} --device {args.device}",
            f"Train {args.stage} detector"):
            sys.exit(1)
        
        logger.info("\n" + "="*70)
        logger.info("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("  1. Validate: python train_orchestrator.py --validate best.pt")
        logger.info("  2. Export:   python train_orchestrator.py --export best.pt")
        logger.info("  3. Deploy:   Update solution.py with trained model path")
        return
    
    # Individual steps
    if args.download:
        if not run_command("python download_github_datasets.py --all",
                         "Download GitHub datasets"):
            sys.exit(1)
    
    if args.prepare:
        if not run_command("python prepare_datasets.py --all",
                         "Prepare datasets"):
            sys.exit(1)
    
    if args.train:
        if not run_command(
            f"python train_quick.py --stage {args.stage} "
            f"--data prepared_datasets/final_dataset/data.yaml "
            f"--epochs {args.epochs} --batch {args.batch} --device {args.device}",
            f"Train {args.stage}"):
            sys.exit(1)
    
    if args.validate:
        if not run_command(
            f"python train_quick.py --validate {args.validate} "
            f"--data prepared_datasets/final_dataset/data.yaml",
            "Validate model"):
            sys.exit(1)
    
    if args.export:
        if not run_command(
            f"python train_quick.py --export {args.export} --format onnx",
            "Export model to ONNX"):
            sys.exit(1)
    
    if args.tensorboard:
        logger.info("Starting TensorBoard...")
        logger.info("Open: http://localhost:6006")
        os.system("tensorboard --logdir training_output")
    
    if args.summary:
        run_command("python download_github_datasets.py --summary",
                   "Show dataset summary")
        run_command("python prepare_datasets.py --summary",
                   "Show prepared dataset summary")
    
    if args.clean:
        logger.info("Cleaning output directories...")
        for dir_path in ["datasets", "prepared_datasets", "training_output"]:
            if Path(dir_path).exists():
                import shutil
                shutil.rmtree(dir_path)
                logger.info(f"Removed: {dir_path}")
    
    # If no arguments, show help
    if not any([args.full_pipeline, args.quick_test, args.download, args.prepare,
                args.train, args.validate, args.export, args.tensorboard,
                args.summary, args.clean]):
        parser.print_help()
        logger.info("\n" + "="*70)
        logger.info("RECOMMENDED WORKFLOWS")
        logger.info("="*70)
        logger.info("\n1. QUICK TEST (5-10 minutes):")
        logger.info("   python train_orchestrator.py --quick-test\n")
        logger.info("2. FULL TRAINING (2-3 hours with GPU):")
        logger.info("   python train_orchestrator.py --full-pipeline\n")
        logger.info("3. STEP-BY-STEP:")
        logger.info("   python train_orchestrator.py --download")
        logger.info("   python train_orchestrator.py --prepare")
        logger.info("   python train_orchestrator.py --train --stage motorcycle")


if __name__ == "__main__":
    main()
