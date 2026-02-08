"""
Configuration file for kidney stone detection pipeline.
Contains hyperparameters, paths, and system settings.
"""

import os
from pathlib import Path

class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "datasets"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    LOG_DIR = BASE_DIR / "logs"
    EXPORT_DIR = BASE_DIR / "exported_models"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, EXPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Dataset settings
    IMAGE_SIZE = (512, 512)  # Standard size for medical imaging
    NUM_CLASSES = 2  # Stone present or absent
    PIXEL_SPACING = 0.5  # mm per pixel (typical for CT scans)
    
    # Imaging modality support
    SUPPORTED_MODALITIES = ['CT', 'Ultrasound', 'X-ray']
    DEFAULT_MODALITY = 'CT'
    
    # Data augmentation parameters
    AUGMENTATION_CONFIG = {
        'rotation_range': 15,  # degrees
        'zoom_range': 0.15,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2],
        'contrast_range': [0.8, 1.2],
        'elastic_transform': True,
        'gaussian_noise': 0.01
    }
    
    # Preprocessing settings
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    DENOISE_H = 10  # Denoising strength
    NORMALIZATION_METHOD = 'standardize'  # 'standardize' or 'min-max'
    
    # Model architecture
    BACKBONE = 'efficientnet-b3'  # Options: resnet50, efficientnet-b3, densenet121
    USE_PRETRAINED = True
    DETECTION_METHOD = 'yolov8'  # Options: yolov8, faster-rcnn, unet
    
    # Multi-task learning
    CLASSIFICATION_WEIGHT = 1.0
    REGRESSION_WEIGHT = 0.5
    SEGMENTATION_WEIGHT = 0.8  # If using segmentation
    
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = 'adamw'  # Options: adamw, sgd
    
    # Learning rate schedule
    LR_SCHEDULER = 'cosine'  # Options: cosine, step, exponential
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-7
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Cross-validation
    NUM_FOLDS = 5
    STRATIFIED = True
    
    # Loss function parameters
    FOCAL_LOSS_GAMMA = 2.0  # For handling class imbalance
    FOCAL_LOSS_ALPHA = 0.25
    SMOOTH_L1_BETA = 1.0  # For size regression
    
    # Evaluation thresholds
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    SIZE_ERROR_TOLERANCE = 2.0  # mm
    
    # Stone size categories (clinical relevance)
    SIZE_CATEGORIES = {
        'small': (0, 5),      # < 5mm
        'medium': (5, 10),    # 5-10mm
        'large': (10, float('inf'))  # > 10mm
    }
    
    # Device settings
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Regularization
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Gradient clipping
    CLIP_GRAD_NORM = 1.0
    
    # Inference settings
    TTA_ENABLED = True  # Test-time augmentation
    TTA_TRANSFORMS = ['original', 'horizontal_flip', 'rotate_5', 'rotate_-5']
    
    # Explainability
    GRADCAM_LAYER = 'layer4'  # Target layer for Grad-CAM
    GENERATE_SALIENCY = True
    
    # Export formats
    EXPORT_FORMATS = ['onnx', 'torchscript']
    ONNX_OPSET = 14
    
    # Medical compliance
    ANONYMIZE_DATA = True
    AUDIT_LOGGING = True
    MIN_CONFIDENCE_FOR_DEPLOYMENT = 0.85
    
    # Dataset sources (public datasets)
    DATASET_SOURCES = {
        'KiTS': 'https://kits19.grand-challenge.org/',
        'CT_Kidney': 'https://wiki.cancerimagingarchive.net/',
        'TCIA': 'https://www.cancerimagingarchive.net/'
    }
    
    @classmethod
    def get_device(cls):
        """Determine computation device"""
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("Kidney Stone Detection Pipeline Configuration")
        print("=" * 60)
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Backbone: {cls.BACKBONE}")
        print(f"Detection Method: {cls.DETECTION_METHOD}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Device: {cls.get_device()}")
        print("=" * 60)
