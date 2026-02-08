# kidney_stone_detection
This is a complete, production-ready machine learning pipeline for detecting kidney stones in medical imaging with automatic size estimation.
## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Preparation

### Expected Data Format

The pipeline expects annotations in CSV or JSON format:

```json
{
  "image_id": "patient_001_scan_01",
  "image_path": "images/patient_001_scan_01.dcm",
  "has_stone": 1,
  "stone_bbox": [120, 180, 45, 45],
  "stone_size_mm": 7.5,
  "modality": "CT",
  "patient_id": "patient_001",
  "scan_date": "2024-01-15"
}
```

### Public Dataset Sources

1. **KiTS Challenge Dataset** (Kidney Tumor Segmentation)
   - URL: https://kits19.grand-challenge.org/
   - Contains CT scans of kidneys with annotations
   - Can be adapted for stone detection

2. **Cancer Imaging Archive (TCIA)**
   - URL: https://www.cancerimagingarchive.net/
   - Large collection of medical images including kidney CT scans

3. **NIH Clinical Center Dataset**
   - Various medical imaging datasets
   - Requires institutional access

### Data Organization

```
datasets/
├── train/
│   ├── images/
│   │   ├── patient_001_scan_01.dcm
│   │   ├── patient_002_scan_01.png
│   │   └── ...
│   └── annotations.csv
├── val/
│   ├── images/
│   └── annotations.csv
└── test/
    ├── images/
    └── annotations.csv
```

### Data Preprocessing

```python
from data.preprocessing import MedicalImagePreprocessor
from utils.config import Config

config = Config()
preprocessor = MedicalImagePreprocessor(config)

# Preprocess a single image
image = preprocessor.load_image("path/to/image.dcm", modality="CT")
processed = preprocessor.preprocess_pipeline(
    image,
    apply_denoising=True,
    apply_contrast=True,
    segment_kidney=True
)
```

## Training

### Quick Start

```bash
python training/trainer.py
```

### Custom Training

```python
from training.trainer import Trainer
from data.dataset import create_dataloaders
from models.architecture import build_model
from utils.config import Config

# Configuration
config = Config()
config.BATCH_SIZE = 16
config.NUM_EPOCHS = 100
config.LEARNING_RATE = 1e-4

# Create dataloaders
train_loader, val_loader, _ = create_dataloaders(
    config,
    train_annotations='datasets/train/annotations.csv',
    val_annotations='datasets/val/annotations.csv'
)

# Build model
model = build_model(config)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()
```

### Training with Cross-Validation

```python
from data.dataset import create_stratified_folds

# Create 5-fold cross-validation splits
create_stratified_folds(
    annotations_file='datasets/all_annotations.csv',
    config=config,
    output_dir='datasets/folds'
)

# Train on each fold
for fold in range(5):
    print(f"\nTraining fold {fold + 1}/5")
    
    train_loader, val_loader, _ = create_dataloaders(
        config,
        train_annotations=f'datasets/folds/fold_{fold}_train.csv',
        val_annotations=f'datasets/folds/fold_{fold}_val.csv'
    )
    
    model = build_model(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
```

### Monitoring Training

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir=logs/
```

## Inference

### Single Image Prediction

```python
from inference.predictor import KidneyStonePredictor
from utils.config import Config

config = Config()
predictor = KidneyStonePredictor(
    model_path="checkpoints/best_model.pth",
    config=config,
    use_gradcam=True
)

# Make prediction
results = predictor.predict(
    image_path="path/to/test/image.dcm",
    modality="CT",
    return_visualization=True
)

# Print results
print(f"Stone detected: {results['has_stone']}")
print(f"Confidence: {results['confidence']:.2%}")
if results['has_stone']:
    print(f"Size: {results['stone_size_mm']:.1f} mm")
    print(f"Category: {results['size_category']}")

# Generate report
report = predictor.generate_report(results)
print(report)

# Visualize with Grad-CAM
predictor.visualize_prediction(results, save_path="prediction.png")
```

### Batch Prediction

```python
# Predict on multiple images
image_paths = [
    "test/image1.dcm",
    "test/image2.dcm",
    "test/image3.dcm"
]

results = predictor.predict_batch(image_paths)

for result in results:
    print(f"\n{result['image_path']}:")
    print(f"  Stone: {result['has_stone']}")
    print(f"  Size: {result.get('stone_size_mm', 0):.1f} mm")
```

## Evaluation

### Compute Metrics

```python
from utils.metrics import MetricsTracker
from utils.config import Config

config = Config()
metrics = MetricsTracker(config)

# During validation loop
for batch in val_loader:
    outputs = model.predict(batch['image'])
    targets = {
        'label': batch['label'],
        'size': batch['size']
    }
    metrics.update(outputs, targets)

# Print comprehensive results
metrics.print_summary()

# Get specific metrics
results = metrics.compute_all()
print(f"F1-Score: {results['detection']['f1_score']:.4f}")
print(f"MAE: {results['regression']['mae']:.4f} mm")
print(f"Category Accuracy: {results['clinical']['category_accuracy']:.2f}%")
```

### Clinical Evaluation

```python
# Size category performance
size_metrics = metrics.compute_size_category_metrics()
for category, perf in size_metrics.items():
    print(f"{category}: MAE = {perf['mae']:.2f} mm")

# False negative analysis
fn_analysis = metrics.compute_false_negative_analysis()
print(f"Missed stones: {fn_analysis['count']}")
print(f"Mean size of missed: {fn_analysis['mean_size']:.2f} mm")
```

## Model Deployment

### Export to ONNX

```bash
python scripts/deploy.py
```

Or programmatically:

```python
from scripts.deploy import ModelExporter
from utils.config import Config

config = Config()
exporter = ModelExporter("checkpoints/best_model.pth", config)

# Export to ONNX
onnx_path = exporter.export_onnx()

# Export to TorchScript
torchscript_path = exporter.export_torchscript()

# Create deployment package
package = exporter.create_deployment_package()
```

### Production Inference with ONNX

```python
import onnxruntime
import numpy as np

# Load ONNX model
session = onnxruntime.InferenceSession("exported_models/kidney_stone_detector.onnx")

# Prepare input
image = preprocess_image("test_image.dcm")
input_dict = {'image': image}

# Run inference
outputs = session.run(None, input_dict)
logits, size, features = outputs

# Parse results
probabilities = softmax(logits)
has_stone = probabilities[0, 1] > 0.5
stone_size = size[0, 0]
```

### REST API Deployment

Example Flask server:

```python
from flask import Flask, request, jsonify
from inference.predictor import KidneyStonePredictor
import tempfile
import os

app = Flask(__name__)
predictor = KidneyStonePredictor("checkpoints/best_model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Predict
        result = predictor.predict(tmp_path, return_visualization=False)
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Configuration

Key configuration parameters in `utils/config.py`:

```python
# Image settings
IMAGE_SIZE = (512, 512)
PIXEL_SPACING = 0.5  # mm per pixel

# Model architecture
BACKBONE = 'efficientnet-b3'
USE_PRETRAINED = True

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Loss weights
CLASSIFICATION_WEIGHT = 1.0
REGRESSION_WEIGHT = 0.5

# Evaluation
CONFIDENCE_THRESHOLD = 0.5
SIZE_ERROR_TOLERANCE = 2.0  # mm
```

## Performance Optimization

### Mixed Precision Training

```python
config.USE_MIXED_PRECISION = True  # Enabled by default
```

### Data Loading

```python
config.NUM_WORKERS = 4  # Adjust based on CPU cores
config.PIN_MEMORY = True  # For faster GPU transfer
```

### Gradient Accumulation

```python
# For effective batch size of 32 with batch_size=8
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Medical AI Ethics and Compliance

### Data Privacy

- **Anonymization**: All patient IDs are hashed
- **Audit Logging**: All data access is logged
- **HIPAA Compliance**: Follow your institution's guidelines

### Regulatory Considerations

This software is intended for **research purposes only**. For clinical deployment:

1. **FDA Approval**: Medical AI systems require FDA clearance (510(k) or De Novo)
2. **CE Marking**: Required for European markets
3. **Clinical Validation**: Prospective clinical studies required
4. **Quality Management**: ISO 13485 compliance
5. **Risk Assessment**: ISO 14971 medical device risk management

### Human-in-the-Loop

**Critical**: This system is designed to **assist**, not replace, radiologists:

- All predictions should be reviewed by qualified medical professionals
- The model provides a "second opinion" to reduce oversight
- Clinical decisions should never rely solely on AI predictions
- Regular audit of model performance in production

### Bias and Fairness

- Model performance may vary across demographics
- Validate on diverse patient populations
- Monitor for performance degradation over time
- Implement continuous learning pipelines

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
config.BATCH_SIZE = 8

# Enable mixed precision
config.USE_MIXED_PRECISION = True

# Reduce image size
config.IMAGE_SIZE = (384, 384)
```

**2. Slow Training**
```python
# Increase data loading workers
config.NUM_WORKERS = 8

# Use smaller backbone
config.BACKBONE = 'efficientnet-b0'
```

**3. Poor Detection Performance**
```python
# Adjust loss weights
config.CLASSIFICATION_WEIGHT = 2.0

# Use focal loss for class imbalance
# Already implemented in MultiTaskLoss

# Increase data augmentation
config.AUGMENTATION_CONFIG['rotation_range'] = 20
```

**4. Inaccurate Size Estimation**
```python
# Increase regression loss weight
config.REGRESSION_WEIGHT = 1.0

# Use weighted MSE for size categories
# Modify in losses.py to use WeightedMSELoss
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows PEP 8 style
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{kidney_stone_detection,
  title={Kidney Stone Detection and Size Estimation Pipeline},
  author={Your Institution},
  year={2024},
  url={https://github.com/your-repo/kidney-stone-detection}
}
```

## License

This project is licensed under the MIT License for research purposes.

**For clinical use, additional regulatory approval is required.**

## Acknowledgments

- Pretrained models from the timm library
- Medical imaging community for open datasets
- Radiologists for clinical validation

## Contact

For questions or collaboration:
- Email: medical-ai@institution.edu
- Issues: GitHub issue tracker

---

⚠️ **DISCLAIMER**: This software is for research and educational purposes only. It is NOT approved for clinical use. Always consult with qualified medical professionals for diagnosis and treatment decisions.
