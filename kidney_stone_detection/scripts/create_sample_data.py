#!/usr/bin/env python
"""
Utility script to generate sample/dummy data for testing the pipeline.
Creates synthetic medical images with annotations for development and testing.

Usage:
    python scripts/create_sample_data.py --num-samples 100 --output datasets/sample
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm


def create_synthetic_kidney_image(size=(512, 512), has_stone=False, stone_size=0):
    """
    Create a synthetic medical image resembling a kidney CT scan.
    
    Args:
        size: Image dimensions (H, W)
        has_stone: Whether to include a kidney stone
        stone_size: Size of stone in millimeters
    
    Returns:
        Tuple of (image, bbox) where bbox is [x, y, w, h] or None
    """
    # Create base image (simulating tissue)
    image = np.random.normal(50, 15, size).astype(np.float32)
    image = np.clip(image, 0, 255)
    
    # Add kidney region (darker)
    h, w = size
    center_x, center_y = w // 2 + np.random.randint(-50, 50), h // 2 + np.random.randint(-50, 50)
    kidney_radius = np.random.randint(120, 180)
    
    y, x = np.ogrid[:h, :w]
    kidney_mask = (x - center_x)**2 + (y - center_y)**2 <= kidney_radius**2
    image[kidney_mask] = np.random.normal(30, 10, np.sum(kidney_mask))
    
    # Add some texture
    noise = np.random.normal(0, 5, size)
    image = image + noise
    image = np.clip(image, 0, 255)
    
    # Smooth
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    bbox = None
    if has_stone:
        # Add kidney stone (bright spot)
        # Convert size from mm to pixels (assuming 0.5 mm/pixel)
        stone_radius_pixels = int(stone_size / 0.5 / 2)
        stone_radius_pixels = max(5, stone_radius_pixels)  # Minimum 5 pixels
        
        # Position stone inside kidney
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, kidney_radius * 0.7)
        stone_x = int(center_x + distance * np.cos(angle))
        stone_y = int(center_y + distance * np.sin(angle))
        
        # Create stone
        y, x = np.ogrid[:h, :w]
        stone_mask = (x - stone_x)**2 + (y - stone_y)**2 <= stone_radius_pixels**2
        
        # Make stone bright (calcium appears bright in CT)
        image[stone_mask] = np.random.normal(180, 10, np.sum(stone_mask))
        
        # Add subtle shadow/artifact
        shadow_mask = (x - stone_x)**2 + (y - stone_y)**2 <= (stone_radius_pixels * 1.5)**2
        shadow_mask = shadow_mask & ~stone_mask
        image[shadow_mask] = image[shadow_mask] * 0.9
        
        # Create bounding box
        bbox = [
            max(0, stone_x - stone_radius_pixels * 2),
            max(0, stone_y - stone_radius_pixels * 2),
            stone_radius_pixels * 4,
            stone_radius_pixels * 4
        ]
    
    # Add scanning artifacts
    for _ in range(np.random.randint(2, 5)):
        line_x = np.random.randint(0, w)
        line_intensity = np.random.normal(0, 2)
        image[:, line_x] += line_intensity
    
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image, bbox


def generate_sample_dataset(num_samples, output_dir, positive_ratio=0.6):
    """
    Generate a sample dataset with synthetic images.
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory path
        positive_ratio: Ratio of positive (stone present) cases
    """
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    
    print(f"Generating {num_samples} synthetic medical images...")
    
    for i in tqdm(range(num_samples)):
        # Decide if this sample has a stone
        has_stone = np.random.rand() < positive_ratio
        
        # Generate stone size if present
        if has_stone:
            # Sample from realistic distribution
            # Small stones (< 5mm): 40%
            # Medium stones (5-10mm): 40%
            # Large stones (> 10mm): 20%
            category = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
            
            if category == 'small':
                stone_size = np.random.uniform(2, 5)
            elif category == 'medium':
                stone_size = np.random.uniform(5, 10)
            else:
                stone_size = np.random.uniform(10, 20)
        else:
            stone_size = 0
        
        # Create image
        image, bbox = create_synthetic_kidney_image(
            size=(512, 512),
            has_stone=has_stone,
            stone_size=stone_size
        )
        
        # Save image
        image_filename = f'sample_{i:04d}.png'
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), image)
        
        # Create annotation
        annotation = {
            'image_id': f'sample_{i:04d}',
            'image_path': f'images/{image_filename}',
            'has_stone': int(has_stone),
            'stone_size_mm': round(stone_size, 2),
            'modality': 'CT',
            'patient_id': f'patient_{i:03d}',
            'scan_date': f'2024-01-{(i % 28) + 1:02d}'
        }
        
        if bbox is not None:
            annotation['stone_bbox'] = bbox
        
        annotations.append(annotation)
    
    # Save annotations
    annotations_df = pd.DataFrame(annotations)
    
    # Save as CSV
    csv_path = output_path / 'annotations.csv'
    annotations_df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = output_path / 'annotations.json'
    with open(json_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nDataset created successfully!")
    print(f"Images saved to: {images_dir}")
    print(f"Annotations saved to: {csv_path} and {json_path}")
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(annotations_df)}")
    print(f"  Positive cases: {annotations_df['has_stone'].sum()}")
    print(f"  Negative cases: {(1 - annotations_df['has_stone']).sum()}")
    print(f"  Mean stone size: {annotations_df[annotations_df['has_stone']==1]['stone_size_mm'].mean():.2f} mm")


def create_train_val_test_split(base_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        base_dir: Directory containing annotations.csv
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test gets remainder)
    """
    base_path = Path(base_dir)
    annotations_path = base_path / 'annotations.csv'
    
    if not annotations_path.exists():
        print(f"Error: Annotations not found at {annotations_path}")
        return
    
    # Load annotations
    df = pd.read_csv(annotations_path)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Save splits
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = base_path.parent / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Copy images directory link (or you could copy actual images)
        split_csv = split_dir / 'annotations.csv'
        split_df.to_csv(split_csv, index=False)
        
        print(f"{split_name.capitalize()} set: {len(split_df)} samples")
        print(f"  Saved to: {split_csv}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate sample dataset for testing'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/sample',
        help='Output directory for generated data'
    )
    
    parser.add_argument(
        '--positive-ratio',
        type=float,
        default=0.6,
        help='Ratio of positive (stone present) cases'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Automatically split into train/val/test sets'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Generate dataset
    generate_sample_dataset(
        num_samples=args.num_samples,
        output_dir=args.output,
        positive_ratio=args.positive_ratio
    )
    
    # Split if requested
    if args.split:
        print("\nCreating train/val/test splits...")
        create_train_val_test_split(args.output)
    
    print("\n" + "="*70)
    print("Sample data creation completed!")
    print("="*70)
    print("\nYou can now train the model using:")
    print(f"  python train.py --train-annotations {args.output}/train/annotations.csv \\")
    print(f"                  --val-annotations {args.output}/val/annotations.csv")


if __name__ == '__main__':
    main()
