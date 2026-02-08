"""
Image preprocessing module for medical imaging data.
Handles normalization, enhancement, and kidney segmentation.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import exposure, filters
import pydicom
from typing import Tuple, Optional, Union
import warnings

class MedicalImagePreprocessor:
    """
    Preprocesses medical images (CT, X-ray, Ultrasound) for kidney stone detection.
    """
    
    def __init__(self, config):
        self.config = config
        self.image_size = config.IMAGE_SIZE
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_SIZE
        )
    
    def load_dicom(self, filepath: str) -> np.ndarray:
        """
        Load DICOM file and extract pixel array.
        Handles Hounsfield units for CT scans.
        """
        try:
            dicom = pydicom.dcmread(filepath)
            image = dicom.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept (HU units for CT)
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                image = image * dicom.RescaleSlope + dicom.RescaleIntercept
            
            # Window for kidney stone visibility (typical CT window)
            # Stone window: Level=50, Width=400
            window_center = 50
            window_width = 400
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2
            image = np.clip(image, img_min, img_max)
            
            return image
        
        except Exception as e:
            warnings.warn(f"Error loading DICOM: {e}")
            return None
    
    def load_image(self, filepath: str, modality: str = 'CT') -> np.ndarray:
        """
        Load medical image from various formats.
        Supports DICOM, PNG, JPG, TIFF.
        """
        if filepath.lower().endswith('.dcm'):
            image = self.load_dicom(filepath)
        else:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {filepath}")
            image = image.astype(np.float32)
        
        return image
    
    def denoise_image(self, image: np.ndarray, method: str = 'nlm') -> np.ndarray:
        """
        Apply denoising to reduce image noise while preserving edges.
        """
        if method == 'nlm':
            # Non-local means denoising - good for medical images
            image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(image_uint8, None, 
                                                 h=self.config.DENOISE_H,
                                                 templateWindowSize=7,
                                                 searchWindowSize=21)
            return denoised.astype(np.float32)
        
        elif method == 'bilateral':
            # Bilateral filter - preserves edges
            image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            denoised = cv2.bilateralFilter(image_uint8, 9, 75, 75)
            return denoised.astype(np.float32)
        
        elif method == 'gaussian':
            # Simple gaussian blur
            return cv2.GaussianBlur(image, (5, 5), 0)
        
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Essential for enhancing stone visibility in medical images.
        """
        # Normalize to uint8 for CLAHE
        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced = self.clahe.apply(image_uint8)
        return enhanced.astype(np.float32)
    
    def normalize_image(self, image: np.ndarray, method: str = 'standardize') -> np.ndarray:
        """
        Normalize image intensities.
        """
        if method == 'standardize':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            return image
        
        elif method == 'min-max':
            # Min-max normalization to [0, 1]
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            return image
        
        return image
    
    def segment_kidney_region(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roughly segment kidney region to reduce search space.
        Uses intensity-based thresholding and morphological operations.
        
        Note: For production, consider using a pretrained kidney segmentation model.
        """
        # Normalize for processing
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply gaussian blur
        blurred = cv2.GaussianBlur(img_norm, (5, 5), 0)
        
        # Otsu's thresholding to separate tissue
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours and keep significant regions
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(image)
        if contours:
            # Keep largest contours (assumed to be kidneys)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            cv2.drawContours(mask, sorted_contours, -1, 1, -1)
        
        # Apply mask to original image
        masked_image = image * mask
        
        return masked_image, mask
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio with padding.
        """
        if target_size is None:
            target_size = self.image_size
        
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    
    def preprocess_pipeline(self, 
                           image: np.ndarray,
                           apply_denoising: bool = True,
                           apply_contrast: bool = True,
                           segment_kidney: bool = False) -> dict:
        """
        Complete preprocessing pipeline.
        Returns dictionary with processed image and intermediate results.
        """
        results = {'original': image.copy()}
        
        # Step 1: Denoising
        if apply_denoising:
            image = self.denoise_image(image)
            results['denoised'] = image.copy()
        
        # Step 2: Contrast enhancement
        if apply_contrast:
            image = self.enhance_contrast(image)
            results['enhanced'] = image.copy()
        
        # Step 3: Kidney segmentation (optional)
        if segment_kidney:
            image, mask = self.segment_kidney_region(image)
            results['segmentation_mask'] = mask
            results['masked'] = image.copy()
        
        # Step 4: Resize
        image = self.resize_image(image)
        results['resized'] = image.copy()
        
        # Step 5: Normalization
        image = self.normalize_image(image, method=self.config.NORMALIZATION_METHOD)
        results['normalized'] = image
        
        # Final output
        results['final'] = image
        
        return results
    
    def batch_preprocess(self, images: list, **kwargs) -> np.ndarray:
        """
        Preprocess a batch of images.
        """
        processed = []
        for img in images:
            result = self.preprocess_pipeline(img, **kwargs)
            processed.append(result['final'])
        
        return np.stack(processed)


class DataAugmenter:
    """
    Medical image augmentation with domain-specific transformations.
    """
    
    def __init__(self, config):
        self.config = config
        self.aug_config = config.AUGMENTATION_CONFIG
    
    def random_rotation(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Rotate image by random angle"""
        angle = np.random.uniform(-self.aug_config['rotation_range'], 
                                   self.aug_config['rotation_range'])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        if mask is not None:
            rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
            return rotated_img, rotated_mask
        
        return rotated_img, None
    
    def random_zoom(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply random zoom"""
        zoom_factor = np.random.uniform(1 - self.aug_config['zoom_range'],
                                        1 + self.aug_config['zoom_range'])
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize
        zoomed_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            zoomed_img = zoomed_img[start_h:start_h+h, start_w:start_w+w]
            if mask is not None:
                zoomed_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                zoomed_mask = zoomed_mask[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            zoomed_img = cv2.copyMakeBorder(zoomed_img, pad_h, h-new_h-pad_h, 
                                            pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
            if mask is not None:
                zoomed_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                zoomed_mask = cv2.copyMakeBorder(zoomed_mask, pad_h, h-new_h-pad_h,
                                                  pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
        
        if mask is not None:
            return zoomed_img, zoomed_mask
        return zoomed_img, None
    
    def random_flip(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Horizontal flip"""
        if self.aug_config['horizontal_flip'] and np.random.rand() > 0.5:
            flipped_img = cv2.flip(image, 1)
            if mask is not None:
                flipped_mask = cv2.flip(mask, 1)
                return flipped_img, flipped_mask
            return flipped_img, None
        return image, mask
    
    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast"""
        brightness = np.random.uniform(*self.aug_config['brightness_range'])
        contrast = np.random.uniform(*self.aug_config['contrast_range'])
        
        # Adjust
        adjusted = image * contrast + (brightness - 1) * np.mean(image)
        return np.clip(adjusted, image.min(), image.max())
    
    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add gaussian noise to simulate imaging artifacts"""
        if self.aug_config['gaussian_noise'] > 0:
            noise = np.random.normal(0, self.aug_config['gaussian_noise'], image.shape)
            noisy = image + noise
            return np.clip(noisy, image.min(), image.max())
        return image
    
    def elastic_transform(self, image: np.ndarray, alpha=100, sigma=10) -> np.ndarray:
        """
        Elastic deformation for medical image augmentation.
        Simulates tissue deformation.
        """
        if not self.aug_config['elastic_transform']:
            return image
        
        if np.random.rand() > 0.5:
            return image
        
        shape = image.shape
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        return ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    def augment(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """
        Apply full augmentation pipeline.
        """
        # Geometric transformations
        image, mask = self.random_rotation(image, mask)
        image, mask = self.random_zoom(image, mask)
        image, mask = self.random_flip(image, mask)
        
        # Intensity transformations (only on image)
        image = self.adjust_brightness_contrast(image)
        image = self.add_gaussian_noise(image)
        image = self.elastic_transform(image)
        
        return image, mask
