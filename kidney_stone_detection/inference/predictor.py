"""
Inference pipeline for kidney stone detection.
Includes prediction, visualization, and explainability (Grad-CAM).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

from data.preprocessing import MedicalImagePreprocessor
from models.architecture import build_model
from utils.config import Config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for explainability.
    Shows which regions of the image influenced the model's decision.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        target = None
        for name, module in self.model.named_modules():
            if name == self.target_layer or self.target_layer in str(module):
                target = module
                break
        
        if target is None:
            # Use last conv layer of backbone
            for module in self.model.backbone.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target = module
        
        if target:
            target.register_forward_hook(forward_hook)
            target.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image [1, C, H, W]
            target_class: Target class for visualization (1 for stone)
        
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output['logits'][0, target_class]
        class_score.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            warnings.warn("Gradients or activations not captured")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Pool gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
        
        # Average over channels
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        
        # ReLU to keep only positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


class KidneyStonePredictor:
    """
    Complete inference pipeline for kidney stone detection.
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Config = None,
                 use_gradcam: bool = True):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
            use_gradcam: Whether to generate Grad-CAM visualizations
        """
        if config is None:
            config = Config()
        
        self.config = config
        self.device = config.get_device()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = MedicalImagePreprocessor(config)
        
        # Grad-CAM
        self.gradcam = None
        if use_gradcam:
            try:
                self.gradcam = GradCAM(self.model, config.GRADCAM_LAYER)
            except Exception as e:
                warnings.warn(f"Failed to initialize Grad-CAM: {e}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = build_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Loaded model from {model_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs")
        
        return model
    
    def predict(self, 
                image_path: str,
                modality: str = 'CT',
                return_visualization: bool = True) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to medical image
            modality: Imaging modality (CT, Ultrasound, X-ray)
            return_visualization: Whether to generate visualization
        
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path, modality)
        processed = self.preprocessor.preprocess_pipeline(
            image,
            apply_denoising=True,
            apply_contrast=True,
            segment_kidney=False
        )
        
        # Prepare input tensor
        image_tensor = torch.from_numpy(processed['final']).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict(
                image_tensor,
                threshold=self.config.CONFIDENCE_THRESHOLD
            )
        
        # Extract results
        has_stone = bool(predictions['predictions'][0].item())
        confidence = float(predictions['probabilities'][0].item())
        stone_size = float(predictions['size'][0].item())
        
        # Determine size category
        size_category = self._categorize_size(stone_size)
        
        # Generate Grad-CAM if requested
        heatmap = None
        if return_visualization and self.gradcam is not None and has_stone:
            try:
                heatmap = self.gradcam.generate(image_tensor, target_class=1)
            except Exception as e:
                warnings.warn(f"Failed to generate Grad-CAM: {e}")
        
        results = {
            'has_stone': has_stone,
            'confidence': confidence,
            'stone_size_mm': stone_size if has_stone else 0.0,
            'size_category': size_category if has_stone else 'none',
            'original_image': processed['original'],
            'processed_image': processed['final'],
            'heatmap': heatmap
        }
        
        return results
    
    def _categorize_size(self, size: float) -> str:
        """Categorize stone size"""
        for category, (min_size, max_size) in self.config.SIZE_CATEGORIES.items():
            if min_size <= size < max_size:
                return category
        return 'unknown'
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_visualization=False)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, 
                            results: Dict,
                            save_path: Optional[str] = None) -> None:
        """
        Create visualization of prediction with Grad-CAM overlay.
        
        Args:
            results: Prediction results dictionary
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(results['original_image'], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(results['processed_image'], cmap='gray')
        axes[1].set_title('Preprocessed Image')
        axes[1].axis('off')
        
        # Prediction with Grad-CAM
        axes[2].imshow(results['processed_image'], cmap='gray')
        
        if results['heatmap'] is not None:
            # Resize heatmap to match image size
            heatmap = cv2.resize(
                results['heatmap'],
                (results['processed_image'].shape[1], results['processed_image'].shape[0])
            )
            
            # Apply colormap
            heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
            
            # Overlay
            axes[2].imshow(heatmap_colored, alpha=0.5)
        
        # Add prediction text
        prediction_text = f"Stone Detected: {results['has_stone']}\n"
        prediction_text += f"Confidence: {results['confidence']:.2%}\n"
        if results['has_stone']:
            prediction_text += f"Size: {results['stone_size_mm']:.1f} mm\n"
            prediction_text += f"Category: {results['size_category']}"
        
        axes[2].set_title(prediction_text, fontsize=10)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate clinical report from prediction.
        
        Args:
            results: Prediction results
        
        Returns:
            Formatted text report
        """
        report = "="*60 + "\n"
        report += "KIDNEY STONE DETECTION REPORT\n"
        report += "="*60 + "\n\n"
        
        report += "FINDINGS:\n"
        report += "-"*60 + "\n"
        
        if results['has_stone']:
            report += f"Stone detected with {results['confidence']:.1%} confidence.\n"
            report += f"Estimated size: {results['stone_size_mm']:.1f} mm\n"
            report += f"Size category: {results['size_category'].upper()}\n\n"
            
            report += "CLINICAL SIGNIFICANCE:\n"
            report += "-"*60 + "\n"
            
            size = results['stone_size_mm']
            if size < 5:
                report += "Small stone (< 5mm): High likelihood of spontaneous passage.\n"
                report += "Conservative management recommended.\n"
            elif size < 10:
                report += "Medium stone (5-10mm): Possible spontaneous passage.\n"
                report += "Medical intervention may be required.\n"
            else:
                report += "Large stone (> 10mm): Low likelihood of spontaneous passage.\n"
                report += "Surgical intervention likely required.\n"
        else:
            report += f"No kidney stone detected (confidence: {1-results['confidence']:.1%}).\n"
        
        report += "\n"
        report += "="*60 + "\n"
        report += "NOTE: This is an AI-assisted analysis.\n"
        report += "Clinical correlation and radiologist review recommended.\n"
        report += "="*60 + "\n"
        
        return report


def demo_inference():
    """
    Demo function showing how to use the predictor.
    """
    # Configuration
    config = Config()
    
    # Initialize predictor
    model_path = "checkpoints/best_model.pth"
    predictor = KidneyStonePredictor(model_path, config)
    
    # Example prediction
    image_path = "path/to/test/image.dcm"
    
    if Path(image_path).exists():
        # Make prediction
        results = predictor.predict(image_path)
        
        # Print report
        report = predictor.generate_report(results)
        print(report)
        
        # Visualize
        predictor.visualize_prediction(results, save_path="prediction_result.png")
    else:
        print(f"Test image not found: {image_path}")
        print("Please provide a valid image path")


if __name__ == '__main__':
    demo_inference()
