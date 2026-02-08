#!/usr/bin/env python
"""
Evaluation script for kidney stone detection model.
Computes comprehensive metrics on test set and generates visualizations.

Usage:
    python evaluate.py --model checkpoints/best_model.pth --data datasets/test/annotations.csv
    python evaluate.py --model checkpoints/best_model.pth --data datasets/test/annotations.csv --visualize
"""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference.predictor import KidneyStonePredictor
from utils.metrics import MetricsTracker
from utils.config import Config
from data.dataset import KidneyStoneDataset
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate kidney stone detection model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to test annotations file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save individual predictions to CSV'
    )
    
    return parser.parse_args()


def evaluate_model(model_path, test_loader, config):
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to model checkpoint
        test_loader: Test data loader
        config: Configuration object
    
    Returns:
        MetricsTracker with computed metrics
    """
    # Load model
    device = config.get_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    from models.architecture import build_model
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)
    
    # Store predictions for later analysis
    all_predictions = []
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            sizes = batch['size'].to(device)
            
            # Get predictions
            predictions = model.predict(images, threshold=config.CONFIDENCE_THRESHOLD)
            
            # Update metrics
            targets = {
                'label': labels,
                'size': sizes
            }
            metrics_tracker.update(predictions, targets)
            
            # Store predictions
            for i in range(len(images)):
                pred_dict = {
                    'image_id': batch['image_id'][i],
                    'true_label': labels[i].item(),
                    'predicted_label': predictions['predictions'][i].item(),
                    'confidence': predictions['probabilities'][i].item(),
                    'true_size': sizes[i].item(),
                    'predicted_size': predictions['size'][i].item()
                }
                all_predictions.append(pred_dict)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    return metrics_tracker, all_predictions


def create_visualizations(metrics_tracker, output_dir):
    """Create and save evaluation visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    print("  - Confusion matrix")
    metrics_tracker.detection_metrics.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    # Size prediction scatter plot
    print("  - Size prediction plot")
    metrics_tracker.regression_metrics.plot_predictions(
        save_path=output_dir / 'size_predictions.png'
    )
    
    # ROC curve (if probabilities available)
    if len(metrics_tracker.detection_metrics.probabilities) > 0:
        print("  - ROC curve")
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(
            metrics_tracker.detection_metrics.targets,
            metrics_tracker.detection_metrics.probabilities
        )
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Size distribution by category
    print("  - Size distribution")
    results = metrics_tracker.compute_all()
    category_metrics = results['regression_by_category']
    
    if category_metrics:
        categories = list(category_metrics.keys())
        maes = [category_metrics[cat]['mae'] for cat in categories]
        rmses = [category_metrics[cat]['rmse'] for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE by category
        ax1.bar(categories, maes, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Mean Absolute Error (mm)')
        ax1.set_title('Size Estimation MAE by Category')
        ax1.grid(True, alpha=0.3)
        
        # RMSE by category
        ax2.bar(categories, rmses, color='coral', alpha=0.7)
        ax2.set_ylabel('Root Mean Squared Error (mm)')
        ax2.set_title('Size Estimation RMSE by Category')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def save_results(metrics_tracker, predictions, output_dir):
    """Save evaluation results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving results...")
    
    # Compute all metrics
    results = metrics_tracker.compute_all()
    
    # Save metrics to JSON
    metrics_file = output_dir / 'metrics.json'
    
    # Convert to serializable format
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    with open(metrics_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"  - Metrics saved to {metrics_file}")
    
    # Save predictions to CSV
    if predictions:
        predictions_file = output_dir / 'predictions.csv'
        df = pd.DataFrame(predictions)
        df.to_csv(predictions_file, index=False)
        print(f"  - Predictions saved to {predictions_file}")
    
    # Save text report
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("KIDNEY STONE DETECTION MODEL - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("DETECTION METRICS (Classification)\n")
        f.write("-"*70 + "\n")
        for key, value in results['detection'].items():
            if isinstance(value, float):
                f.write(f"{key:25s}: {value:.4f}\n")
            else:
                f.write(f"{key:25s}: {value}\n")
        
        f.write("\n")
        f.write("SIZE ESTIMATION METRICS (Regression)\n")
        f.write("-"*70 + "\n")
        for key, value in results['regression'].items():
            f.write(f"{key:25s}: {value:.4f}\n")
        
        f.write("\n")
        f.write("SIZE CATEGORY PERFORMANCE\n")
        f.write("-"*70 + "\n")
        for category, metrics in results['regression_by_category'].items():
            f.write(f"\n{category.upper()}:\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key:20s}: {value:.4f}\n")
                else:
                    f.write(f"  {key:20s}: {value}\n")
        
        f.write("\n")
        f.write("CLINICAL RELEVANCE\n")
        f.write("-"*70 + "\n")
        for key, value in results['clinical'].items():
            if isinstance(value, float):
                f.write(f"{key:25s}: {value:.4f}\n")
            else:
                f.write(f"{key:25s}: {value}\n")
        
        f.write("\n")
        f.write("FALSE NEGATIVE ANALYSIS\n")
        f.write("-"*70 + "\n")
        fn_analysis = results['false_negatives']
        f.write(f"Total missed stones: {fn_analysis['count']}\n")
        if fn_analysis['count'] > 0:
            f.write(f"Mean size of missed stones: {fn_analysis.get('mean_size', 0):.2f} mm\n")
            f.write(f"Size distribution: {fn_analysis['size_distribution']}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"  - Report saved to {report_file}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return
    
    # Check if data exists
    if not Path(args.data).exists():
        print(f"Error: Test data not found at {args.data}")
        return
    
    # Load configuration
    config = Config()
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    print("Configuration:")
    config.print_config()
    
    # Create dataset and dataloader
    print(f"\nLoading test data from {args.data}")
    test_dataset = KidneyStoneDataset(
        data_dir=config.DATA_DIR,
        annotations_file=args.data,
        config=config,
        mode='test',
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Evaluate
    metrics_tracker, predictions = evaluate_model(args.model, test_loader, config)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    metrics_tracker.print_summary()
    
    # Save results
    save_results(metrics_tracker, predictions if args.save_predictions else None, args.output_dir)
    
    # Create visualizations
    if args.visualize:
        create_visualizations(metrics_tracker, args.output_dir)
    
    print("\n" + "="*70)
    print("Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    import numpy as np
    main()
