# Dataset Preparation Guide

This guide explains how to prepare medical imaging data for the kidney stone detection pipeline.

## Table of Contents
1. [Dataset Requirements](#dataset-requirements)
2. [Data Collection](#data-collection)
3. [Annotation Format](#annotation-format)
4. [Data Organization](#data-organization)
5. [Privacy and Compliance](#privacy-and-compliance)
6. [Quality Control](#quality-control)
7. [Data Augmentation](#data-augmentation)

## Dataset Requirements

### Minimum Dataset Size

For acceptable performance:
- **Minimum**: 500 images (300 train, 100 val, 100 test)
- **Recommended**: 2,000+ images (1,400 train, 300 val, 300 test)
- **Optimal**: 10,000+ images for production-grade models

### Class Balance

Aim for reasonable class distribution:
- **Positive cases (stones present)**: 40-70%
- **Negative cases (no stones)**: 30-60%

For size distribution (among positive cases):
- Small stones (< 5mm): 30-40%
- Medium stones (5-10mm): 40-50%
- Large stones (> 10mm): 10-30%

### Image Quality Requirements

- **Resolution**: Minimum 256x256, recommended 512x512 or higher
- **Format**: DICOM (.dcm), PNG, JPEG, or TIFF
- **Bit depth**: 8-bit or 16-bit grayscale
- **Modality**: CT (preferred), X-ray, or Ultrasound

## Data Collection

### Public Datasets

#### 1. KiTS Challenge (Kidney Tumor Segmentation)
- **URL**: https://kits19.grand-challenge.org/
- **Content**: 300+ kidney CT scans with segmentation
- **Usage**: Can be adapted for stone detection training
- **Access**: Free registration required

#### 2. The Cancer Imaging Archive (TCIA)
- **URL**: https://www.cancerimagingarchive.net/
- **Content**: Large collection of medical imaging
- **Collections**:
  - CT Colonography
  - Kidney and Kidney Tumor Segmentation
  - Various CT scan databases
- **Access**: Free, some require approval

#### 3. NIH Clinical Center Datasets
- **Content**: Various medical imaging datasets
- **Access**: Requires institutional access
- **Quality**: High-quality clinical data

### Institutional Data

If using data from your institution:

1. **Ethics Approval**: Obtain IRB/ethics committee approval
2. **Patient Consent**: Ensure proper informed consent
3. **Anonymization**: Remove all patient identifiers
4. **Data Sharing Agreement**: Follow institutional policies

### Data Acquisition Guidelines

#### For CT Scans:
- **Slice thickness**: 1-5mm
- **Field of view**: Entire kidney region visible
- **Contrast**: Both contrast and non-contrast acceptable
- **Phase**: Any acquisition phase (arterial, venous, delayed)

#### For X-rays:
- **View**: KUB (Kidneys, Ureters, Bladder) preferred
- **Quality**: Clear visualization of kidney areas
- **Positioning**: Standard radiographic positioning

#### For Ultrasound:
- **Views**: Longitudinal and transverse
- **Settings**: B-mode imaging
- **Quality**: Clear kidney parenchyma visible

## Annotation Format

### CSV Format

The pipeline expects annotations in CSV format with these columns:

```csv
image_id,image_path,has_stone,stone_size_mm,stone_bbox,modality,patient_id,scan_date
patient_001_scan_01,images/patient_001_scan_01.dcm,1,7.5,"[120,180,45,45]",CT,patient_001,2024-01-15
patient_002_scan_01,images/patient_002_scan_01.dcm,0,0.0,,CT,patient_002,2024-01-16
```

### JSON Format

Alternatively, use JSON:

```json
[
  {
    "image_id": "patient_001_scan_01",
    "image_path": "images/patient_001_scan_01.dcm",
    "has_stone": 1,
    "stone_size_mm": 7.5,
    "stone_bbox": [120, 180, 45, 45],
    "modality": "CT",
    "patient_id": "patient_001",
    "scan_date": "2024-01-15"
  }
]
```

### Required Fields

- `image_id` (string): Unique identifier for the image
- `image_path` (string): Relative path to image file
- `has_stone` (int): 1 if stone present, 0 otherwise
- `stone_size_mm` (float): Stone size in millimeters (0 if no stone)

### Optional Fields

- `stone_bbox` (list): Bounding box [x, y, width, height]
- `modality` (string): Imaging modality (CT, X-ray, Ultrasound)
- `patient_id` (string): Anonymized patient identifier
- `scan_date` (string): Date of scan acquisition

### Annotation Guidelines

#### Measuring Stone Size

1. **Linear Measurement**: Measure longest diameter
2. **Units**: Always in millimeters (mm)
3. **Precision**: Round to 1 decimal place
4. **Multiple Stones**: Record largest stone size

#### Creating Bounding Boxes

Format: `[x, y, width, height]`
- `x`: Left coordinate
- `y`: Top coordinate
- `width`: Box width in pixels
- `height`: Box height in pixels

Guidelines:
- Include stone with small margin (2-5mm)
- Square boxes preferred for circular stones
- Tight fit for irregular shapes

#### Quality Assurance

For each annotation:
- [ ] Image loads correctly
- [ ] Stone size measured accurately
- [ ] Bounding box contains stone
- [ ] No patient identifiers visible
- [ ] Correct modality specified

## Data Organization

### Directory Structure

```
datasets/
├── train/
│   ├── images/
│   │   ├── patient_001_scan_01.dcm
│   │   ├── patient_002_scan_01.dcm
│   │   └── ...
│   └── annotations.csv
├── val/
│   ├── images/
│   │   └── ...
│   └── annotations.csv
└── test/
    ├── images/
    │   └── ...
    └── annotations.csv
```

### Splitting Strategy

#### Random Split
```python
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(data, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.176, random_state=42)
# Results in 70% train, 15% val, 15% test
```

#### Stratified Split
```python
from data.dataset import create_stratified_folds

create_stratified_folds(
    annotations_file='all_annotations.csv',
    config=config,
    output_dir='datasets/folds'
)
```

#### Patient-level Split

**Critical**: Ensure all images from the same patient are in the same split!

```python
# Group by patient
patient_ids = df['patient_id'].unique()

# Split patients
train_patients, test_patients = train_test_split(
    patient_ids, test_size=0.15, random_state=42
)

# Filter data
train_df = df[df['patient_id'].isin(train_patients)]
test_df = df[df['patient_id'].isin(test_patients)]
```

## Privacy and Compliance

### HIPAA Compliance

For US healthcare data:

1. **De-identification**:
   - Remove all 18 HIPAA identifiers
   - Hash patient IDs
   - Remove dates (or shift consistently)
   - Remove location data

2. **Audit Trails**:
   - Log all data access
   - Maintain access controls
   - Document data usage

3. **Security**:
   - Encrypt data at rest
   - Secure transmission
   - Access controls

### Anonymization Script

```python
import hashlib
import pandas as pd

def anonymize_dataset(input_csv, output_csv):
    """Anonymize patient identifiers"""
    df = pd.read_csv(input_csv)
    
    # Hash patient IDs
    df['patient_id'] = df['patient_id'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
    )
    
    # Remove or shift dates
    # Option 1: Remove
    if 'scan_date' in df.columns:
        df = df.drop(columns=['scan_date'])
    
    # Option 2: Shift consistently
    # date_shift = pd.Timedelta(days=random.randint(1, 365))
    # df['scan_date'] = pd.to_datetime(df['scan_date']) + date_shift
    
    # Rename images to remove patient info
    df['image_id'] = ['img_' + str(i).zfill(6) for i in range(len(df))]
    
    df.to_csv(output_csv, index=False)
    print(f"Anonymized data saved to {output_csv}")

# Usage
anonymize_dataset('raw_annotations.csv', 'anonymized_annotations.csv')
```

### GDPR Compliance

For European data:
- Right to erasure: Implement data deletion
- Data minimization: Collect only necessary data
- Consent management: Track consent status
- Data portability: Export in standard format

## Quality Control

### Image Quality Checks

Run these checks on all images:

```python
import cv2
import numpy as np
from pathlib import Path

def check_image_quality(image_path):
    """Check if image meets quality standards"""
    try:
        # Load image
        if image_path.suffix == '.dcm':
            import pydicom
            dcm = pydicom.dcmread(image_path)
            image = dcm.pixel_array
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return False, "Failed to load"
        
        # Check dimensions
        h, w = image.shape[:2]
        if h < 256 or w < 256:
            return False, f"Too small: {h}x{w}"
        
        # Check if image is blank
        if np.std(image) < 5:
            return False, "Image too uniform (possibly blank)"
        
        # Check dynamic range
        if image.max() - image.min() < 50:
            return False, "Insufficient contrast"
        
        return True, "OK"
    
    except Exception as e:
        return False, str(e)

# Usage
for image_path in Path('datasets/train/images').glob('*'):
    is_valid, message = check_image_quality(image_path)
    if not is_valid:
        print(f"Issue with {image_path.name}: {message}")
```

### Annotation Quality Checks

```python
def validate_annotations(annotations_file):
    """Validate annotation file"""
    df = pd.read_csv(annotations_file)
    
    issues = []
    
    # Check required columns
    required = ['image_id', 'image_path', 'has_stone', 'stone_size_mm']
    missing = set(required) - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for duplicates
    duplicates = df['image_id'].duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate image_ids")
    
    # Check stone sizes
    positive_cases = df[df['has_stone'] == 1]
    if len(positive_cases) > 0:
        if (positive_cases['stone_size_mm'] <= 0).any():
            issues.append("Positive cases with zero/negative size")
        
        if (positive_cases['stone_size_mm'] > 50).any():
            issues.append("Suspicious large stone sizes (> 50mm)")
    
    # Check negative cases
    negative_cases = df[df['has_stone'] == 0]
    if (negative_cases['stone_size_mm'] != 0).any():
        issues.append("Negative cases with non-zero size")
    
    if issues:
        print("Annotation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All annotations valid")
        return True
```

## Data Augmentation

The pipeline includes built-in augmentation, but you can customize:

### Augmentation Configuration

```python
from utils.config import Config

config = Config()

# Customize augmentation
config.AUGMENTATION_CONFIG = {
    'rotation_range': 20,        # degrees
    'zoom_range': 0.2,          # 20% zoom
    'horizontal_flip': True,
    'brightness_range': [0.7, 1.3],
    'contrast_range': [0.7, 1.3],
    'elastic_transform': True,
    'gaussian_noise': 0.02
}
```

### Best Practices

1. **Geometric Augmentations**:
   - Rotation: ±15-20 degrees
   - Zoom: ±10-20%
   - Flipping: Horizontal only (vertical can alter anatomy)

2. **Intensity Augmentations**:
   - Brightness: ±20-30%
   - Contrast: ±20-30%
   - Gaussian noise: σ = 0.01-0.02

3. **Medical-Specific**:
   - Elastic deformation (simulates tissue movement)
   - CLAHE application variations
   - Simulated imaging artifacts

4. **What NOT to augment**:
   - Don't use vertical flips (anatomy-dependent)
   - Avoid extreme rotations (> 30 degrees)
   - Don't change stone size artificially

## Dataset Statistics

Track these metrics for your dataset:

```python
def compute_dataset_statistics(annotations_file):
    """Compute and display dataset statistics"""
    df = pd.read_csv(annotations_file)
    
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Positive cases: {df['has_stone'].sum()} ({df['has_stone'].mean()*100:.1f}%)")
    print(f"Negative cases: {(1-df['has_stone']).sum()} ({(1-df['has_stone'].mean())*100:.1f}%)")
    
    if df['has_stone'].sum() > 0:
        positive_df = df[df['has_stone'] == 1]
        print(f"\nStone Size Statistics:")
        print(f"  Mean: {positive_df['stone_size_mm'].mean():.2f} mm")
        print(f"  Median: {positive_df['stone_size_mm'].median():.2f} mm")
        print(f"  Std: {positive_df['stone_size_mm'].std():.2f} mm")
        print(f"  Min: {positive_df['stone_size_mm'].min():.2f} mm")
        print(f"  Max: {positive_df['stone_size_mm'].max():.2f} mm")
        
        print(f"\nSize Distribution:")
        small = (positive_df['stone_size_mm'] < 5).sum()
        medium = ((positive_df['stone_size_mm'] >= 5) & 
                 (positive_df['stone_size_mm'] < 10)).sum()
        large = (positive_df['stone_size_mm'] >= 10).sum()
        
        print(f"  Small (< 5mm): {small} ({small/len(positive_df)*100:.1f}%)")
        print(f"  Medium (5-10mm): {medium} ({medium/len(positive_df)*100:.1f}%)")
        print(f"  Large (> 10mm): {large} ({large/len(positive_df)*100:.1f}%)")
```

## Troubleshooting

### Common Issues

**Issue**: Images won't load
- Check file permissions
- Verify file paths are correct
- Ensure DICOM library installed (pydicom)

**Issue**: Poor model performance
- Check class balance (add augmentation if imbalanced)
- Verify annotation quality
- Increase dataset size
- Check for duplicate images

**Issue**: Memory errors
- Reduce image size
- Decrease batch size
- Use data streaming instead of loading all

## Further Resources

- [DICOM Standard](https://www.dicomstandard.org/)
- [Medical Image Processing Libraries](https://github.com/topics/medical-imaging)
- [Data Annotation Tools](https://github.com/topics/annotation-tool)

---

For questions about dataset preparation, consult the main README.md or open an issue.
