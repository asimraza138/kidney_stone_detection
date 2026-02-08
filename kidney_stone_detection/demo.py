#!/usr/bin/env python
"""
Quick Start Demo - End-to-End Pipeline Demonstration
This script demonstrates the complete workflow from data generation to inference.

Usage:
    python demo.py
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}")
        print(f"Error: {e}")
        return False


def demo_pipeline():
    """Run complete demo pipeline"""
    
    print("\n" + "="*70)
    print("  KIDNEY STONE DETECTION - QUICK START DEMO")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Generate sample synthetic data")
    print("  2. Train a model (in debug mode - fast)")
    print("  3. Evaluate the model")
    print("  4. Run inference on test images")
    print("  5. Export the model for deployment")
    
    input("\nPress Enter to start the demo...")
    
    # Step 1: Generate sample data
    print_section("STEP 1: Generating Sample Dataset")
    print("Creating 200 synthetic medical images with annotations...")
    print("This simulates real medical imaging data for development.\n")
    
    success = run_command(
        ['python', 'scripts/create_sample_data.py', 
         '--num-samples', '200',
         '--output', 'datasets/demo',
         '--split'],
        "Data generation"
    )
    
    if not success:
        print("Demo failed at data generation step.")
        return
    
    # Step 2: Train model
    print_section("STEP 2: Training Model")
    print("Training a model in DEBUG mode (fast, for demonstration)...")
    print("For real training, remove the --debug flag.\n")
    
    success = run_command(
        ['python', 'train.py',
         '--train-annotations', 'datasets/demo/train/annotations.csv',
         '--val-annotations', 'datasets/demo/val/annotations.csv',
         '--debug',
         '--epochs', '3',
         '--batch-size', '4'],
        "Model training"
    )
    
    if not success:
        print("Demo failed at training step.")
        return
    
    # Step 3: Evaluate model
    print_section("STEP 3: Evaluating Model")
    print("Computing comprehensive metrics on test set...\n")
    
    # Check if model exists
    model_path = Path('checkpoints/best_model.pth')
    if not model_path.exists():
        print("Warning: Best model not found, looking for latest checkpoint...")
        checkpoints = list(Path('checkpoints').glob('*.pth'))
        if checkpoints:
            model_path = sorted(checkpoints)[-1]
            print(f"Using: {model_path}")
        else:
            print("No checkpoints found. Skipping evaluation.")
            model_path = None
    
    if model_path:
        success = run_command(
            ['python', 'evaluate.py',
             '--model', str(model_path),
             '--data', 'datasets/demo/test/annotations.csv',
             '--output-dir', 'demo_results',
             '--visualize',
             '--save-predictions'],
            "Model evaluation"
        )
    
    # Step 4: Run inference
    print_section("STEP 4: Running Inference")
    print("Making predictions on test images with Grad-CAM visualization...\n")
    
    if model_path:
        print("To run inference on your own images, use:")
        print(f"  python -c \"")
        print(f"from inference.predictor import KidneyStonePredictor")
        print(f"from utils.config import Config")
        print(f"predictor = KidneyStonePredictor('{model_path}', Config())")
        print(f"result = predictor.predict('path/to/image.dcm')")
        print(f"print(predictor.generate_report(result))")
        print(f"predictor.visualize_prediction(result)")
        print(f"  \"")
        print()
    
    # Step 5: Export model
    print_section("STEP 5: Exporting Model for Deployment")
    print("Converting model to ONNX format for production use...\n")
    
    if model_path:
        success = run_command(
            ['python', 'scripts/deploy.py'],
            "Model export"
        )
    
    # Summary
    print_section("DEMO COMPLETED!")
    
    print("What was created:")
    print("  ✓ Sample dataset with 200 images")
    print("  ✓ Trained model checkpoint")
    print("  ✓ Evaluation results and visualizations")
    print("  ✓ Exported ONNX model for deployment")
    
    print("\nNext steps:")
    print("  1. Review results in demo_results/")
    print("  2. Check exported model in exported_models/")
    print("  3. Use your own medical imaging data")
    print("  4. Train with full dataset (remove --debug)")
    print("  5. Deploy model to production")
    
    print("\nFor real use:")
    print("  - Replace sample data with actual medical images")
    print("  - Train for 50-100 epochs without --debug")
    print("  - Validate with radiologist review")
    print("  - Follow regulatory guidelines for clinical use")
    
    print("\nDocumentation:")
    print("  - README.md: Complete usage guide")
    print("  - Check individual scripts for detailed options")
    print("  - See requirements.txt for dependencies")
    
    print("\n" + "="*70)
    print("Thank you for trying the Kidney Stone Detection pipeline!")
    print("="*70 + "\n")


def quick_inference_example():
    """Show a quick inference example"""
    print_section("Quick Inference Example")
    
    example_code = '''
from inference.predictor import KidneyStonePredictor
from utils.config import Config

# Initialize predictor
config = Config()
predictor = KidneyStonePredictor(
    model_path="checkpoints/best_model.pth",
    config=config
)

# Predict on a single image
result = predictor.predict("path/to/medical/image.dcm")

# Print results
print(f"Stone detected: {result['has_stone']}")
print(f"Confidence: {result['confidence']:.2%}")
if result['has_stone']:
    print(f"Size: {result['stone_size_mm']:.1f} mm")
    print(f"Category: {result['size_category']}")

# Generate clinical report
report = predictor.generate_report(result)
print(report)

# Visualize with Grad-CAM
predictor.visualize_prediction(result, save_path="prediction.png")

# Batch prediction
image_paths = ["img1.dcm", "img2.dcm", "img3.dcm"]
results = predictor.predict_batch(image_paths)
'''
    
    print("Python code for inference:")
    print("-" * 70)
    print(example_code)
    print("-" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run quick start demo')
    parser.add_argument('--example-only', action='store_true',
                       help='Only show inference example without running demo')
    
    args = parser.parse_args()
    
    if args.example_only:
        quick_inference_example()
    else:
        try:
            demo_pipeline()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            print("You can restart the demo anytime by running: python demo.py")
        except Exception as e:
            print(f"\n\nError during demo: {e}")
            import traceback
            traceback.print_exc()
