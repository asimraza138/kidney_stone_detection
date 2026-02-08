"""
Evaluation metrics for kidney stone detection and size estimation.
Includes clinical relevance metrics and standard ML metrics.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DetectionMetrics:
    """
    Metrics for binary classification (stone detection).
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and targets"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: torch.Tensor = None):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Binary predictions [B]
            targets: Ground truth labels [B]
            probabilities: Prediction probabilities [B] (optional)
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all classification metrics.
        
        Returns:
            Dictionary of metric values
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['precision'] = precision_score(targets, preds, zero_division=0)
        metrics['recall'] = recall_score(targets, preds, zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['f1_score'] = f1_score(targets, preds, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Positive and Negative Predictive Values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # ROC-AUC (if probabilities available)
        if len(self.probabilities) > 0:
            try:
                metrics['roc_auc'] = roc_auc_score(targets, self.probabilities)
            except ValueError:
                # Handle case with only one class
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix"""
        return confusion_matrix(self.targets, self.predictions)
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot and optionally save confusion matrix"""
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Stone', 'Stone'],
                   yticklabels=['No Stone', 'Stone'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Stone Detection')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class RegressionMetrics:
    """
    Metrics for size estimation (regression).
    """
    
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset stored values"""
        self.predictions = []
        self.targets = []
        self.masks = []  # To track which samples have stones
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               mask: torch.Tensor = None):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Predicted sizes [B]
            targets: Ground truth sizes [B]
            mask: Binary mask for valid samples [B]
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if mask is not None:
            self.masks.extend(mask.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Returns:
            Dictionary of metric values
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Filter to only positive cases if masks available
        if len(self.masks) > 0:
            masks = np.array(self.masks).astype(bool)
            preds = preds[masks]
            targets = targets[masks]
        
        if len(preds) == 0:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'r2': 0.0
            }
        
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(preds - targets))
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(np.mean((preds - targets) ** 2))
        
        # Mean Absolute Percentage Error
        # Avoid division by zero
        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((preds[non_zero_mask] - targets[non_zero_mask]) / 
                                  targets[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = 0.0
        
        # R² Score
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Clinical relevance: accuracy within tolerance
        tolerance = self.config.SIZE_ERROR_TOLERANCE  # mm
        within_tolerance = np.mean(np.abs(preds - targets) <= tolerance) * 100
        metrics['within_tolerance_pct'] = within_tolerance
        
        return metrics
    
    def compute_by_size_category(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics separately for each size category.
        Clinically relevant for treatment planning.
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if len(self.masks) > 0:
            masks = np.array(self.masks).astype(bool)
            preds = preds[masks]
            targets = targets[masks]
        
        category_metrics = {}
        
        for category, (min_size, max_size) in self.config.SIZE_CATEGORIES.items():
            # Filter by category
            category_mask = (targets >= min_size) & (targets < max_size)
            
            if not np.any(category_mask):
                continue
            
            cat_preds = preds[category_mask]
            cat_targets = targets[category_mask]
            
            category_metrics[category] = {
                'mae': np.mean(np.abs(cat_preds - cat_targets)),
                'rmse': np.sqrt(np.mean((cat_preds - cat_targets) ** 2)),
                'count': int(np.sum(category_mask))
            }
        
        return category_metrics
    
    def plot_predictions(self, save_path: str = None):
        """Plot predicted vs actual sizes"""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if len(self.masks) > 0:
            masks = np.array(self.masks).astype(bool)
            preds = preds[masks]
            targets = targets[masks]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, preds, alpha=0.5)
        
        # Perfect prediction line
        max_val = max(targets.max(), preds.max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        
        # Tolerance bands
        tolerance = self.config.SIZE_ERROR_TOLERANCE
        plt.fill_between([0, max_val], 
                        [0-tolerance, max_val-tolerance],
                        [0+tolerance, max_val+tolerance],
                        alpha=0.2, color='green', label=f'±{tolerance}mm tolerance')
        
        plt.xlabel('Actual Size (mm)')
        plt.ylabel('Predicted Size (mm)')
        plt.title('Kidney Stone Size Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ClinicalMetrics:
    """
    Clinical relevance metrics for medical decision making.
    """
    
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset stored values"""
        self.predictions = []
        self.targets = []
        self.predicted_sizes = []
        self.true_sizes = []
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]):
        """
        Update with predictions and targets.
        
        Args:
            predictions: Dict with 'predictions' and 'size'
            targets: Dict with 'label' and 'size'
        """
        self.predictions.extend(predictions['predictions'].cpu().numpy())
        self.targets.extend(targets['label'].cpu().numpy())
        self.predicted_sizes.extend(predictions['size'].cpu().numpy())
        self.true_sizes.extend(targets['size'].cpu().numpy())
    
    def compute_treatment_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy for treatment-relevant size categories.
        
        Categories:
        - < 5mm: Often pass spontaneously
        - 5-10mm: May require medical intervention
        - > 10mm: Likely require surgical intervention
        """
        preds = np.array(self.predicted_sizes)
        targets = np.array(self.true_sizes)
        
        # Only consider cases where stone is present
        stone_mask = np.array(self.targets) == 1
        preds = preds[stone_mask]
        targets = targets[stone_mask]
        
        if len(preds) == 0:
            return {}
        
        # Categorize predictions and targets
        pred_categories = self._categorize_sizes(preds)
        true_categories = self._categorize_sizes(targets)
        
        # Calculate category agreement
        category_accuracy = np.mean(pred_categories == true_categories) * 100
        
        metrics = {
            'category_accuracy': category_accuracy,
            'correct_small': int(np.sum((pred_categories == 'small') & (true_categories == 'small'))),
            'correct_medium': int(np.sum((pred_categories == 'medium') & (true_categories == 'medium'))),
            'correct_large': int(np.sum((pred_categories == 'large') & (true_categories == 'large')))
        }
        
        return metrics
    
    def _categorize_sizes(self, sizes: np.ndarray) -> np.ndarray:
        """Categorize stone sizes"""
        categories = np.empty(len(sizes), dtype=object)
        
        for category, (min_size, max_size) in self.config.SIZE_CATEGORIES.items():
            mask = (sizes >= min_size) & (sizes < max_size)
            categories[mask] = category
        
        return categories
    
    def compute_false_negative_analysis(self) -> Dict[str, any]:
        """
        Analyze false negative cases (missed stones).
        Critical for patient safety.
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        sizes = np.array(self.true_sizes)
        
        # False negatives: predicted no stone but stone present
        fn_mask = (preds == 0) & (targets == 1)
        
        if not np.any(fn_mask):
            return {'count': 0, 'size_distribution': {}}
        
        fn_sizes = sizes[fn_mask]
        
        # Analyze size distribution of missed stones
        size_dist = {}
        for category, (min_size, max_size) in self.config.SIZE_CATEGORIES.items():
            count = np.sum((fn_sizes >= min_size) & (fn_sizes < max_size))
            size_dist[category] = int(count)
        
        return {
            'count': int(np.sum(fn_mask)),
            'size_distribution': size_dist,
            'mean_size': float(np.mean(fn_sizes)),
            'sizes': fn_sizes.tolist()
        }


class MetricsTracker:
    """
    Central metrics tracker for all evaluation metrics.
    """
    
    def __init__(self, config):
        self.config = config
        self.detection_metrics = DetectionMetrics()
        self.regression_metrics = RegressionMetrics(config)
        self.clinical_metrics = ClinicalMetrics(config)
    
    def reset(self):
        """Reset all metrics"""
        self.detection_metrics.reset()
        self.regression_metrics.reset()
        self.clinical_metrics.reset()
    
    def update(self, outputs: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]):
        """
        Update all metrics with batch outputs.
        
        Args:
            outputs: Model outputs (logits, predictions, probabilities, size)
            targets: Ground truth (label, size)
        """
        # Detection metrics
        self.detection_metrics.update(
            outputs['predictions'],
            targets['label'],
            outputs.get('probabilities')
        )
        
        # Regression metrics
        positive_mask = (targets['label'] == 1)
        if positive_mask.any():
            self.regression_metrics.update(
                outputs['size'],
                targets['size'],
                positive_mask
            )
        
        # Clinical metrics
        self.clinical_metrics.update(outputs, targets)
    
    def compute_all(self) -> Dict[str, any]:
        """Compute all metrics and return as dictionary"""
        results = {
            'detection': self.detection_metrics.compute(),
            'regression': self.regression_metrics.compute(),
            'regression_by_category': self.regression_metrics.compute_by_size_category(),
            'clinical': self.clinical_metrics.compute_treatment_accuracy(),
            'false_negatives': self.clinical_metrics.compute_false_negative_analysis()
        }
        
        return results
    
    def print_summary(self):
        """Print formatted summary of all metrics"""
        results = self.compute_all()
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print("\nDetection Metrics:")
        print("-" * 70)
        for key, value in results['detection'].items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")
            else:
                print(f"  {key:25s}: {value}")
        
        print("\nRegression Metrics (Size Estimation):")
        print("-" * 70)
        for key, value in results['regression'].items():
            print(f"  {key:25s}: {value:.4f}")
        
        print("\nSize Category Performance:")
        print("-" * 70)
        for category, metrics in results['regression_by_category'].items():
            print(f"  {category.capitalize()}:")
            for key, value in metrics.items():
                print(f"    {key:20s}: {value:.4f}" if isinstance(value, float) else f"    {key:20s}: {value}")
        
        print("\nClinical Relevance:")
        print("-" * 70)
        for key, value in results['clinical'].items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")
            else:
                print(f"  {key:25s}: {value}")
        
        print("\nFalse Negative Analysis:")
        print("-" * 70)
        fn_analysis = results['false_negatives']
        print(f"  Total missed stones: {fn_analysis['count']}")
        if fn_analysis['count'] > 0:
            print(f"  Mean size of missed: {fn_analysis.get('mean_size', 0):.2f} mm")
            print(f"  Distribution: {fn_analysis['size_distribution']}")
        
        print("=" * 70 + "\n")
