#!/usr/bin/env python
"""
Main training script for kidney stone detection.
Run this script to train the model from scratch.

Usage:
    python train.py --config configs/default.yaml
    python train.py --resume checkpoints/latest.pth
    python train.py --fold 0  # For k-fold cross-validation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.trainer import Trainer
from data.dataset import create_dataloaders, create_stratified_folds
from models.architecture import build_model, print_model_summary
from utils.config import Config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train kidney stone detection model'
    )
    
    parser.add_argument(
        '--train-annotations',
        type=str,
        default='datasets/train/annotations.csv',
        help='Path to training annotations file'
    )
    
    parser.add_argument(
        '--val-annotations',
        type=str,
        default='datasets/val/annotations.csv',
        help='Path to validation annotations file'
    )
    
    parser.add_argument(
        '--test-annotations',
        type=str,
        default=None,
        help='Path to test annotations file (optional)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Fold number for cross-validation (creates folds if not exist)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default=None,
        choices=['resnet50', 'efficientnet-b0', 'efficientnet-b3', 'densenet121'],
        help='Override backbone architecture'
    )
    
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch without pretrained weights'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode with smaller dataset and fewer epochs'
    )
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration with command line overrides"""
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    
    if args.backbone is not None:
        config.BACKBONE = args.backbone
    
    if args.no_pretrained:
        config.USE_PRETRAINED = False
    
    # Debug mode settings
    if args.debug:
        print("\n*** DEBUG MODE ENABLED ***")
        config.BATCH_SIZE = 4
        config.NUM_EPOCHS = 5
        config.NUM_WORKERS = 0
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Epochs: {config.NUM_EPOCHS}")
        print("***\n")
    
    return config


def setup_cross_validation(args, config):
    """Setup k-fold cross-validation"""
    folds_dir = Path('datasets/folds')
    
    # Check if folds exist
    fold_files = list(folds_dir.glob(f'fold_{args.fold}_*.csv'))
    
    if len(fold_files) == 0:
        print(f"\nFolds not found. Creating {config.NUM_FOLDS}-fold splits...")
        
        # Combine train and val annotations for splitting
        all_annotations = args.train_annotations
        
        create_stratified_folds(
            annotations_file=all_annotations,
            config=config,
            output_dir=str(folds_dir)
        )
    
    # Update annotations paths for this fold
    train_annotations = folds_dir / f'fold_{args.fold}_train.csv'
    val_annotations = folds_dir / f'fold_{args.fold}_val.csv'
    
    print(f"\nUsing fold {args.fold} for cross-validation")
    print(f"Train: {train_annotations}")
    print(f"Val: {val_annotations}")
    
    return str(train_annotations), str(val_annotations)


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    config = setup_config(args)
    config.print_config()
    
    # Handle cross-validation
    train_annotations = args.train_annotations
    val_annotations = args.val_annotations
    
    if args.fold is not None:
        train_annotations, val_annotations = setup_cross_validation(args, config)
    
    # Verify annotation files exist
    if not Path(train_annotations).exists():
        print(f"\nError: Training annotations not found at {train_annotations}")
        print("\nPlease ensure you have:")
        print("1. Downloaded or prepared your dataset")
        print("2. Created annotation files in the expected format")
        print("\nSee README.md for dataset preparation instructions.")
        return
    
    if not Path(val_annotations).exists():
        print(f"\nError: Validation annotations not found at {val_annotations}")
        return
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        test_annotations=args.test_annotations
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    print_model_summary(model)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        if Path(args.resume).exists():
            print(f"\nResuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"\nWarning: Checkpoint not found at {args.resume}")
            print("Starting training from scratch...")
    
    # Train
    try:
        trainer.train()
        
        # Test if test set provided
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_metrics = trainer.validate_on_test(test_loader)
            print("\nTest Set Results:")
            print(f"Loss: {test_loss:.4f}")
            print(f"F1-Score: {test_metrics['detection']['f1_score']:.4f}")
            print(f"MAE: {test_metrics['regression']['mae']:.4f} mm")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(filename='interrupted_checkpoint.pth')
        print("Checkpoint saved. You can resume training with:")
        print(f"  python train.py --resume checkpoints/interrupted_checkpoint.pth")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nSaving checkpoint before exit...")
        try:
            trainer.save_checkpoint(filename='error_checkpoint.pth')
            print("Emergency checkpoint saved.")
        except:
            print("Failed to save emergency checkpoint.")
    
    print("\nTraining script completed.")


if __name__ == '__main__':
    main()
