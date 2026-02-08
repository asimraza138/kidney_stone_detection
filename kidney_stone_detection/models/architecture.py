"""
Deep learning model architectures for kidney stone detection and size estimation.
Implements multi-task learning with classification and regression heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Tuple, Optional


class KidneyStoneDetector(nn.Module):
    """
    Multi-task model for kidney stone detection and size estimation.
    
    Architecture:
    - Backbone: CNN encoder (ResNet, EfficientNet, etc.)
    - Detection head: Binary classification
    - Regression head: Stone size estimation
    - Optional: Segmentation head for precise localization
    """
    
    def __init__(self, config):
        super(KidneyStoneDetector, self).__init__()
        
        self.config = config
        self.backbone_name = config.BACKBONE
        self.num_classes = config.NUM_CLASSES
        
        # Load pretrained backbone
        self.backbone = self._build_backbone()
        
        # Get backbone output features
        self.feature_dim = self._get_feature_dim()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Detection head (classification)
        self.detection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, self.num_classes)
        )
        
        # Size regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 1),
            nn.ReLU()  # Ensure positive size predictions
        )
        
        # Optional: Attention mechanism for feature refinement
        self.attention = SpatialAttention(self.feature_dim)
        
        # Initialize weights
        self._initialize_heads()
    
    def _build_backbone(self):
        """Build CNN backbone using timm library"""
        backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.config.USE_PRETRAINED,
            num_classes=0,  # Remove classification head
            in_chans=1,     # Grayscale medical images
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _get_feature_dim(self):
        """Get output feature dimension of backbone"""
        # Run a dummy forward pass
        dummy_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        return features.shape[1]
    
    def _initialize_heads(self):
        """Initialize classification and regression heads"""
        for module in [self.detection_head, self.regression_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            Dictionary with:
            - logits: Classification logits [B, num_classes]
            - size: Predicted stone size [B, 1]
            - features: Extracted features [B, feature_dim]
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global pooling
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.flatten(1)
        
        # Classification
        logits = self.detection_head(pooled_features)
        
        # Size regression
        size = self.regression_head(pooled_features)
        
        return {
            'logits': logits,
            'size': size,
            'features': pooled_features
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with post-processing.
        
        Args:
            x: Input image tensor
            threshold: Classification threshold
        
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Get probabilities
            probs = F.softmax(outputs['logits'], dim=1)
            stone_prob = probs[:, 1]  # Probability of stone present
            
            # Binary prediction
            predictions = (stone_prob > threshold).long()
            
            # Set size to 0 for negative predictions
            size = outputs['size'].squeeze(-1)
            size = size * predictions.float()
            
            return {
                'predictions': predictions,
                'probabilities': stone_prob,
                'size': size,
                'logits': outputs['logits']
            }


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on relevant regions.
    Helps the model attend to kidney stone locations.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SpatialAttention, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        x = x * spatial_weights
        
        return x


class UNetSegmentation(nn.Module):
    """
    U-Net architecture for kidney stone segmentation.
    Can be used as an alternative or complementary approach.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(UNetSegmentation, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for robust predictions.
    Useful for deployment in critical medical applications.
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Average predictions from all models"""
        all_outputs = [model(x) for model in self.models]
        
        # Weighted average of logits
        logits = torch.stack([out['logits'] for out in all_outputs])
        weighted_logits = (logits * self.weights.view(-1, 1, 1).to(logits.device)).sum(0)
        
        # Weighted average of sizes
        sizes = torch.stack([out['size'] for out in all_outputs])
        weighted_size = (sizes * self.weights.view(-1, 1, 1).to(sizes.device)).sum(0)
        
        return {
            'logits': weighted_logits,
            'size': weighted_size
        }


def build_model(config) -> nn.Module:
    """
    Factory function to build model based on configuration.
    """
    if config.DETECTION_METHOD == 'cnn':
        model = KidneyStoneDetector(config)
    elif config.DETECTION_METHOD == 'unet':
        model = UNetSegmentation()
    else:
        # Default to our multi-task model
        model = KidneyStoneDetector(config)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """Print model architecture summary"""
    print("=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model type: {type(model).__name__}")
    print("=" * 70)
    print(model)
    print("=" * 70)


if __name__ == '__main__':
    # Test model creation
    from utils.config import Config
    
    config = Config()
    model = build_model(config)
    print_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 512, 512)
    outputs = model(dummy_input)
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
