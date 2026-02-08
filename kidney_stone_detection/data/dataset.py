"""
Dataset loader for kidney stone detection.
Handles multi-modal medical imaging data with proper anonymization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import hashlib
from datetime import datetime

from data.preprocessing import MedicalImagePreprocessor, DataAugmenter


class KidneyStoneDataset(Dataset):
    """
    PyTorch dataset for kidney stone detection and size estimation.
    
    Supports:
    - Multiple imaging modalities (CT, Ultrasound, X-ray)
    - Detection (classification + localization)
    - Size estimation (regression)
    """
    
    def __init__(self,
                 data_dir: str,
                 annotations_file: str,
                 config,
                 mode: str = 'train',
                 augment: bool = True,
                 anonymize: bool = True):
        """
        Args:
            data_dir: Directory containing medical images
            annotations_file: JSON or CSV file with annotations
            config: Configuration object
            mode: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            anonymize: Whether to anonymize patient data
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        self.augment = augment and mode == 'train'
        self.anonymize = anonymize
        
        # Initialize preprocessor and augmenter
        self.preprocessor = MedicalImagePreprocessor(config)
        self.augmenter = DataAugmenter(config) if augment else None
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        # Create audit log if required
        if config.AUDIT_LOGGING:
            self._init_audit_log()
    
    def _load_annotations(self, annotations_file: str) -> pd.DataFrame:
        """
        Load and parse annotations file.
        
        Expected format (JSON or CSV):
        {
            "image_id": "patient_001_scan_01",
            "image_path": "images/patient_001_scan_01.dcm",
            "has_stone": 1,
            "stone_bbox": [x, y, width, height],  # Optional
            "stone_size_mm": 7.5,  # Stone size in millimeters
            "modality": "CT",
            "patient_id": "patient_001",  # Will be anonymized
            "scan_date": "2024-01-15"
        }
        """
        file_path = Path(annotations_file)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported annotation format: {file_path.suffix}")
        
        # Anonymize patient IDs
        if self.anonymize and 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].apply(self._anonymize_id)
        
        # Validate required columns
        required_cols = ['image_path', 'has_stone']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle missing stone sizes (set to 0 for negative cases)
        if 'stone_size_mm' not in df.columns:
            df['stone_size_mm'] = 0.0
        df.loc[df['has_stone'] == 0, 'stone_size_mm'] = 0.0
        
        return df
    
    def _anonymize_id(self, patient_id: str) -> str:
        """
        Hash patient ID for anonymization.
        Maintains consistency while protecting privacy.
        """
        return hashlib.sha256(patient_id.encode()).hexdigest()[:16]
    
    def _init_audit_log(self):
        """Initialize audit logging for compliance"""
        self.audit_log = []
    
    def _log_access(self, image_id: str):
        """Log data access for audit trail"""
        if self.config.AUDIT_LOGGING:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'image_id': image_id,
                'mode': self.mode
            }
            self.audit_log.append(log_entry)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - image: Preprocessed image tensor [C, H, W]
            - label: Binary classification label (0 or 1)
            - size: Stone size in mm (0 if no stone)
            - bbox: Bounding box coordinates [x, y, w, h] (if available)
            - image_id: Unique identifier
        """
        # Get annotation
        row = self.annotations.iloc[idx]
        
        # Log access
        image_id = row.get('image_id', f'img_{idx}')
        self._log_access(image_id)
        
        # Load image
        image_path = self.data_dir / row['image_path']
        modality = row.get('modality', self.config.DEFAULT_MODALITY)
        
        try:
            image = self.preprocessor.load_image(str(image_path), modality)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a blank image to avoid training interruption
            image = np.zeros(self.config.IMAGE_SIZE, dtype=np.float32)
        
        # Get labels
        has_stone = int(row['has_stone'])
        stone_size = float(row['stone_size_mm'])
        
        # Get bounding box if available
        bbox = None
        if 'stone_bbox' in row and pd.notna(row['stone_bbox']):
            bbox = self._parse_bbox(row['stone_bbox'])
        
        # Create mask if bbox is available
        mask = None
        if bbox is not None:
            mask = self._create_mask_from_bbox(bbox, image.shape)
        
        # Apply augmentation
        if self.augment and self.augmenter is not None:
            image, mask = self.augmenter.augment(image, mask)
        
        # Preprocess
        processed = self.preprocessor.preprocess_pipeline(
            image,
            apply_denoising=True,
            apply_contrast=True,
            segment_kidney=False  # Can enable for better results
        )
        
        image_tensor = torch.from_numpy(processed['final']).float()
        
        # Add channel dimension if needed
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Prepare output dictionary
        sample = {
            'image': image_tensor,
            'label': torch.tensor(has_stone, dtype=torch.long),
            'size': torch.tensor(stone_size, dtype=torch.float32),
            'image_id': image_id,
            'modality': modality
        }
        
        # Add bbox if available
        if bbox is not None:
            sample['bbox'] = torch.tensor(bbox, dtype=torch.float32)
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            sample['mask'] = mask_tensor
        
        return sample
    
    def _parse_bbox(self, bbox_str) -> List[float]:
        """Parse bounding box from string or list"""
        if isinstance(bbox_str, str):
            # Remove brackets and split
            bbox_str = bbox_str.strip('[]')
            bbox = [float(x) for x in bbox_str.split(',')]
        else:
            bbox = list(bbox_str)
        return bbox
    
    def _create_mask_from_bbox(self, bbox: List[float], shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from bounding box"""
        mask = np.zeros(shape, dtype=np.float32)
        x, y, w, h = [int(v) for v in bbox]
        mask[y:y+h, x:x+w] = 1.0
        return mask
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of positive and negative cases"""
        return {
            'stone_present': int(self.annotations['has_stone'].sum()),
            'no_stone': int((1 - self.annotations['has_stone']).sum())
        }
    
    def get_size_distribution(self) -> Dict[str, int]:
        """Get distribution of stone sizes"""
        positive_cases = self.annotations[self.annotations['has_stone'] == 1]
        
        distribution = {}
        for category, (min_size, max_size) in self.config.SIZE_CATEGORIES.items():
            count = ((positive_cases['stone_size_mm'] >= min_size) & 
                    (positive_cases['stone_size_mm'] < max_size)).sum()
            distribution[category] = int(count)
        
        return distribution
    
    def save_audit_log(self, output_path: str):
        """Save audit log to file"""
        if self.config.AUDIT_LOGGING and self.audit_log:
            with open(output_path, 'w') as f:
                json.dump(self.audit_log, f, indent=2)


def create_dataloaders(config, 
                       train_annotations: str,
                       val_annotations: str,
                       test_annotations: Optional[str] = None) -> Tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        train_annotations: Path to training annotations
        val_annotations: Path to validation annotations
        test_annotations: Path to test annotations (optional)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = KidneyStoneDataset(
        data_dir=config.DATA_DIR,
        annotations_file=train_annotations,
        config=config,
        mode='train',
        augment=True
    )
    
    val_dataset = KidneyStoneDataset(
        data_dir=config.DATA_DIR,
        annotations_file=val_annotations,
        config=config,
        mode='val',
        augment=False
    )
    
    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training class distribution: {train_dataset.get_class_distribution()}")
    print(f"Training size distribution: {train_dataset.get_size_distribution()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = None
    if test_annotations is not None:
        test_dataset = KidneyStoneDataset(
            data_dir=config.DATA_DIR,
            annotations_file=test_annotations,
            config=config,
            mode='test',
            augment=False
        )
        print(f"Test samples: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    
    return train_loader, val_loader, test_loader


def create_stratified_folds(annotations_file: str, 
                            config,
                            output_dir: str):
    """
    Create stratified k-fold splits for cross-validation.
    Ensures balanced distribution of positive/negative cases and stone sizes.
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Load annotations
    if annotations_file.endswith('.json'):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(annotations_file)
    
    # Create stratification labels
    # Combine class and size category for better stratification
    df['size_category'] = pd.cut(
        df['stone_size_mm'],
        bins=[0, 5, 10, float('inf')],
        labels=['small', 'medium', 'large']
    )
    df['strat_label'] = df['has_stone'].astype(str) + '_' + df['size_category'].astype(str)
    
    # Create folds
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_label'])):
        train_df = df.iloc[train_idx].drop(columns=['size_category', 'strat_label'])
        val_df = df.iloc[val_idx].drop(columns=['size_category', 'strat_label'])
        
        # Save fold annotations
        train_file = output_path / f'fold_{fold_idx}_train.csv'
        val_file = output_path / f'fold_{fold_idx}_val.csv'
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        
        print(f"Fold {fold_idx}: Train={len(train_df)}, Val={len(val_df)}")
    
    print(f"Created {config.NUM_FOLDS} folds in {output_dir}")
