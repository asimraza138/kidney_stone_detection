"""
Loss functions for kidney stone detection and size estimation.
Implements multi-task loss with classification and regression components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Especially useful when negative cases (no stone) outnumber positive cases.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [B, num_classes]
            targets: Ground truth labels [B]
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Calculate focal loss
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for size regression.
    Less sensitive to outliers than MSE.
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: Predicted sizes [B, 1] or [B]
            targets: Ground truth sizes [B, 1] or [B]
            mask: Binary mask indicating valid predictions [B]
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # Calculate smooth L1 loss
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Apply mask if provided (only compute loss for positive cases)
        if mask is not None:
            loss = loss * mask.float()
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Useful if using segmentation approach for stone detection.
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted segmentation masks [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    - Classification (stone detection)
    - Regression (size estimation)
    - Optional: Segmentation
    """
    
    def __init__(self, config):
        super(MultiTaskLoss, self).__init__()
        
        self.config = config
        
        # Classification loss
        self.classification_loss = FocalLoss(
            alpha=config.FOCAL_LOSS_ALPHA,
            gamma=config.FOCAL_LOSS_GAMMA
        )
        
        # Regression loss
        self.regression_loss = SmoothL1Loss(beta=config.SMOOTH_L1_BETA)
        
        # Segmentation loss (optional)
        self.segmentation_loss = DiceLoss()
        
        # Loss weights
        self.cls_weight = config.CLASSIFICATION_WEIGHT
        self.reg_weight = config.REGRESSION_WEIGHT
        self.seg_weight = config.SEGMENTATION_WEIGHT
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs containing:
                - logits: Classification logits
                - size: Predicted sizes
                - mask: Predicted segmentation (optional)
            targets: Ground truth containing:
                - label: Classification labels
                - size: True sizes
                - mask: True segmentation (optional)
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Classification loss
        cls_loss = self.classification_loss(outputs['logits'], targets['label'])
        losses['classification'] = cls_loss
        
        # Regression loss (only for positive cases)
        positive_mask = (targets['label'] == 1).float()
        reg_loss = self.regression_loss(
            outputs['size'],
            targets['size'],
            mask=positive_mask
        )
        losses['regression'] = reg_loss
        
        # Segmentation loss (if available)
        if 'mask' in outputs and 'mask' in targets:
            seg_loss = self.segmentation_loss(outputs['mask'], targets['mask'])
            losses['segmentation'] = seg_loss
            total_loss = (self.cls_weight * cls_loss + 
                         self.reg_weight * reg_loss +
                         self.seg_weight * seg_loss)
        else:
            total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        losses['total'] = total_loss
        
        return losses


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss that gives more importance to certain size ranges.
    Useful for ensuring accuracy in clinically relevant size ranges.
    """
    
    def __init__(self, config):
        super(WeightedMSELoss, self).__init__()
        self.config = config
        
        # Define weights for different size categories
        # Higher weights for clinically important boundaries
        self.size_weights = {
            'small': 2.0,   # < 5mm (often pass spontaneously)
            'medium': 3.0,  # 5-10mm (may require intervention)
            'large': 2.0    # > 10mm (likely require intervention)
        }
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted MSE based on size category.
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # Calculate per-sample weights
        weights = torch.ones_like(targets)
        for category, (min_size, max_size) in self.config.SIZE_CATEGORIES.items():
            category_mask = (targets >= min_size) & (targets < max_size)
            weights[category_mask] = self.size_weights[category]
        
        # Compute weighted MSE
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        
        # Apply mask if provided
        if mask is not None:
            weighted_mse = weighted_mse * mask.float()
            return weighted_mse.sum() / (mask.sum() + 1e-8)
        
        return weighted_mse.mean()


class CombinedSizeLoss(nn.Module):
    """
    Combination of MSE and MAE for size estimation.
    MSE penalizes large errors, MAE is more robust to outliers.
    """
    
    def __init__(self, mse_weight: float = 0.5, mae_weight: float = 0.5):
        super(CombinedSizeLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute combined MSE + MAE loss.
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # MSE
        mse = (predictions - targets) ** 2
        
        # MAE
        mae = torch.abs(predictions - targets)
        
        # Combine
        loss = self.mse_weight * mse + self.mae_weight * mae
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


def get_loss_function(config, loss_type: str = 'multi_task'):
    """
    Factory function to get appropriate loss function.
    
    Args:
        config: Configuration object
        loss_type: Type of loss ('multi_task', 'focal', 'smooth_l1', 'dice')
    
    Returns:
        Loss function module
    """
    if loss_type == 'multi_task':
        return MultiTaskLoss(config)
    elif loss_type == 'focal':
        return FocalLoss(config.FOCAL_LOSS_ALPHA, config.FOCAL_LOSS_GAMMA)
    elif loss_type == 'smooth_l1':
        return SmoothL1Loss(config.SMOOTH_L1_BETA)
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(config)
    elif loss_type == 'combined':
        return CombinedSizeLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test loss functions
    from utils.config import Config
    
    config = Config()
    
    # Create dummy data
    batch_size = 4
    num_classes = 2
    
    outputs = {
        'logits': torch.randn(batch_size, num_classes),
        'size': torch.rand(batch_size, 1) * 15  # Random sizes 0-15mm
    }
    
    targets = {
        'label': torch.randint(0, 2, (batch_size,)),
        'size': torch.rand(batch_size) * 15
    }
    
    # Test multi-task loss
    loss_fn = MultiTaskLoss(config)
    losses = loss_fn(outputs, targets)
    
    print("Loss values:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
