"""
Model deployment utilities.
Export trained model to ONNX, TensorFlow Lite, and create deployment package.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import json
import shutil
from typing import Dict, Tuple
import warnings

from models.architecture import build_model
from utils.config import Config


class ModelExporter:
    """
    Export trained model to various formats for deployment.
    """
    
    def __init__(self, model_path: str, config: Config):
        """
        Initialize exporter.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
        """
        self.config = config
        self.device = config.get_device()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Create export directory
        self.export_dir = config.EXPORT_DIR
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = build_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Loaded model from {model_path}")
        return model
    
    def export_onnx(self, 
                    output_filename: str = "kidney_stone_detector.onnx",
                    verify: bool = True) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_filename: Name of output file
            verify: Whether to verify exported model
        
        Returns:
            Path to exported ONNX model
        """
        print("\nExporting model to ONNX format...")
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 1, 
            self.config.IMAGE_SIZE[0],
            self.config.IMAGE_SIZE[1],
            device=self.device
        )
        
        output_path = self.export_dir / output_filename
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.config.ONNX_OPSET,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['logits', 'size', 'features'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'size': {0: 'batch_size'},
                'features': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {output_path}")
        
        # Verify export
        if verify:
            self._verify_onnx_export(output_path, dummy_input)
        
        return str(output_path)
    
    def _verify_onnx_export(self, onnx_path: str, dummy_input: torch.Tensor):
        """Verify ONNX model produces same outputs as PyTorch model"""
        print("\nVerifying ONNX export...")
        
        # Check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure is valid")
        
        # Run inference with PyTorch
        with torch.no_grad():
            pytorch_output = self.model(dummy_input)
        
        # Run inference with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {
            'image': dummy_input.cpu().numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare outputs
        pytorch_logits = pytorch_output['logits'].cpu().numpy()
        onnx_logits = ort_outputs[0]
        
        diff = np.abs(pytorch_logits - onnx_logits).max()
        
        if diff < 1e-5:
            print(f"✓ ONNX export verified (max difference: {diff:.2e})")
        else:
            warnings.warn(f"ONNX outputs differ from PyTorch (max diff: {diff:.2e})")
    
    def export_torchscript(self, output_filename: str = "kidney_stone_detector.pt") -> str:
        """
        Export model to TorchScript format.
        
        Args:
            output_filename: Name of output file
        
        Returns:
            Path to exported model
        """
        print("\nExporting model to TorchScript format...")
        
        # Trace model
        dummy_input = torch.randn(
            1, 1,
            self.config.IMAGE_SIZE[0],
            self.config.IMAGE_SIZE[1],
            device=self.device
        )
        
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save
        output_path = self.export_dir / output_filename
        traced_model.save(str(output_path))
        
        print(f"Model exported to {output_path}")
        return str(output_path)
    
    def create_deployment_package(self, 
                                  package_name: str = "kidney_stone_detector_v1") -> str:
        """
        Create complete deployment package with model, config, and example code.
        
        Args:
            package_name: Name of the deployment package
        
        Returns:
            Path to deployment package
        """
        print("\nCreating deployment package...")
        
        package_dir = self.export_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Export models
        onnx_path = self.export_onnx()
        
        # Copy ONNX model to package
        shutil.copy(onnx_path, package_dir / "model.onnx")
        
        # Save configuration
        config_dict = {
            'image_size': self.config.IMAGE_SIZE,
            'confidence_threshold': self.config.CONFIDENCE_THRESHOLD,
            'size_categories': self.config.SIZE_CATEGORIES,
            'normalization_method': self.config.NORMALIZATION_METHOD,
            'pixel_spacing': self.config.PIXEL_SPACING
        }
        
        with open(package_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Create inference example
        self._create_inference_example(package_dir)
        
        # Create README
        self._create_readme(package_dir)
        
        print(f"Deployment package created at {package_dir}")
        return str(package_dir)
    
    def _create_inference_example(self, package_dir: Path):
        """Create example inference script for deployment"""
        example_code = '''"""
Example inference script for kidney stone detection.
Demonstrates how to use the deployed ONNX model.
"""

import onnxruntime
import numpy as np
import cv2
import json


class KidneyStoneDetectorONNX:
    """
    ONNX-based kidney stone detector for production deployment.
    """
    
    def __init__(self, model_path="model.onnx", config_path="config.json"):
        """Initialize detector with ONNX model"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load ONNX model
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def preprocess(self, image_path):
        """Preprocess medical image"""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        target_size = tuple(self.config['image_size'])
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=(0, 1))
        
        return image
    
    def predict(self, image_path):
        """Make prediction on image"""
        # Preprocess
        image = self.preprocess(image_path)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: image}
        )
        
        # Parse outputs
        logits = outputs[0][0]  # [num_classes]
        size = outputs[1][0][0]  # scalar
        
        # Get prediction
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        stone_probability = probabilities[1]
        has_stone = stone_probability > self.config['confidence_threshold']
        
        return {
            'has_stone': bool(has_stone),
            'confidence': float(stone_probability),
            'stone_size_mm': float(size) if has_stone else 0.0
        }


# Example usage
if __name__ == '__main__':
    detector = KidneyStoneDetectorONNX()
    
    # Make prediction
    result = detector.predict("path/to/image.png")
    
    print(f"Stone detected: {result['has_stone']}")
    print(f"Confidence: {result['confidence']:.2%}")
    if result['has_stone']:
        print(f"Size: {result['stone_size_mm']:.1f} mm")
'''
        
        with open(package_dir / "inference_example.py", 'w') as f:
            f.write(example_code)
    
    def _create_readme(self, package_dir: Path):
        """Create README for deployment package"""
        readme = """# Kidney Stone Detection Model - Deployment Package

## Overview
This package contains a trained deep learning model for detecting kidney stones in medical images and estimating their size.

## Contents
- `model.onnx`: ONNX format model for cross-platform deployment
- `config.json`: Model configuration and parameters
- `inference_example.py`: Example Python script for inference
- `README.md`: This file

## System Requirements
- Python 3.8+
- ONNX Runtime
- NumPy
- OpenCV (for image processing)

## Installation

```bash
pip install onnxruntime numpy opencv-python
```

## Quick Start

```python
from inference_example import KidneyStoneDetectorONNX

# Initialize detector
detector = KidneyStoneDetectorONNX()

# Make prediction
result = detector.predict("path/to/medical/image.png")

print(f"Stone detected: {result['has_stone']}")
print(f"Confidence: {result['confidence']:.2%}")
if result['has_stone']:
    print(f"Estimated size: {result['stone_size_mm']:.1f} mm")
```

## Model Performance
- Detection Accuracy: Refer to validation metrics
- Size Estimation MAE: Refer to validation metrics
- Supported modalities: CT, X-ray, Ultrasound

## Clinical Use
⚠️ **IMPORTANT**: This model is intended to assist radiologists and clinicians. It should NOT be used as a standalone diagnostic tool. Always require human expert review and validation.

## Size Categories
- **Small (< 5mm)**: High likelihood of spontaneous passage
- **Medium (5-10mm)**: May require medical intervention
- **Large (> 10mm)**: Likely requires surgical intervention

## Deployment Considerations
1. **Data Privacy**: Ensure all patient data is anonymized
2. **Regulatory Compliance**: Verify compliance with FDA/CE medical device regulations
3. **Integration**: Work with hospital IT to integrate with PACS systems
4. **Monitoring**: Implement continuous monitoring of model performance
5. **Human-in-the-loop**: Always include radiologist review

## API Integration
The model can be integrated into hospital systems via REST API. Example Flask server:

```python
from flask import Flask, request, jsonify
from inference_example import KidneyStoneDetectorONNX

app = Flask(__name__)
detector = KidneyStoneDetectorONNX()

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    # Save temporarily and predict
    result = detector.predict(image_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Support
For questions or issues, contact the development team.

## License
Refer to your organization's medical software licensing terms.
"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme)


def main():
    """Main deployment function"""
    # Configuration
    config = Config()
    
    # Model path
    model_path = "checkpoints/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Create exporter
    exporter = ModelExporter(model_path, config)
    
    # Export to ONNX
    onnx_path = exporter.export_onnx()
    
    # Export to TorchScript
    torchscript_path = exporter.export_torchscript()
    
    # Create deployment package
    package_path = exporter.create_deployment_package()
    
    print("\n" + "="*70)
    print("Deployment artifacts created successfully!")
    print("="*70)
    print(f"ONNX model: {onnx_path}")
    print(f"TorchScript model: {torchscript_path}")
    print(f"Deployment package: {package_path}")
    print("="*70)


if __name__ == '__main__':
    from utils.config import Config
    main()
