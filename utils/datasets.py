# utils/datasets.py 
"""
Dataset Implementation for Head Pose Estimation

This module implements the AFLW2000 dataset loader following the Deep-Head-Pose methodology.

DATASET OVERVIEW:
- AFLW2000: 2000 face images with 3D head pose annotations
- Annotations: Stored in .mat files with pose parameters and 2D landmarks
- Pose representation: Euler angles (yaw, pitch, roll) in radians, converted to degrees

DATA PREPROCESSING PIPELINE:
1. Face detection and cropping using 2D landmarks
2. Loose cropping with 20% padding for context
3. Image normalization using ImageNet statistics
4. Angle binning: Convert continuous angles to discrete classes

AUGMENTATION STRATEGY:
- Horizontal flipping (50% probability) with angle sign adjustment
- Blur augmentation (5% probability) for robustness

DUAL REPRESENTATION:
- Binned labels: For classification loss (66 bins from -99° to +99°)
- Continuous labels: For regression loss and evaluation metrics
"""
#run with python -m utils.datasets
import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter

from utils.utils import get_ypr_from_mat, get_pt2d_from_mat

def get_list_from_filenames(file_path):
    """
    Load list of filenames from text file
    
    Args:
        file_path: Path to .txt file containing image filenames (one per line)
        
    Returns:
        List of filename strings without extensions
    """
    # input: relative path to .txt file with file names
    # output: list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class AFLW2000(Dataset):
    """
    AFLW2000 Dataset for Head Pose Estimation
    
    This dataset class loads AFLW2000 images and pose annotations following
    the Deep-Head-Pose methodology for consistent evaluation.
    
    Key Features:
    - Face cropping using 2D landmarks with 20% padding
    - Angle conversion from radians to degrees
    - Dual label representation: binned + continuous
    
    Data Flow:
    1. Load image and annotation (.mat file)
    2. Extract 2D landmarks for face cropping
    3. Apply loose cropping with padding
    4. Convert pose angles from radians to degrees
    5. Bin angles into discrete classes [0-65]
    6. Apply image transformations
    """
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        """
        Initialize AFLW2000 dataset
        
        Args:
            data_dir: Directory containing images and .mat annotation files
            filename_path: Path to .txt file with image filenames
            transform: Torchvision transforms to apply to images
            img_ext: Image file extension (default: '.jpg')
            annot_ext: Annotation file extension (default: '.mat')
            image_mode: PIL image mode (default: 'RGB')
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        """
        Get a single data sample
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (image_tensor, binned_labels, continuous_labels, filename)
            - image_tensor: Preprocessed image [3, 224, 224]
            - binned_labels: Angle bins [3] for classification loss
            - continuous_labels: Continuous angles [3] in degrees for regression
            - filename: Original filename for debugging
            
        Processing Pipeline:
        1. Load and convert image to RGB
        2. Extract 2D landmarks from .mat file
        3. Calculate loose face bounding box with 20% padding
        4. Crop face region from image
        5. Load pose angles and convert radians → degrees
        6. Bin angles into 66 discrete classes (-99° to +99°)
        7. Apply image transformations
        """
        # Load image and convert to specified mode (usually RGB)
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Face cropping using 2D landmarks from .mat file
        # This ensures we focus on the face region for pose estimation
        pt2d = get_pt2d_from_mat(mat_path)

        # Calculate bounding box from 2D landmarks
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # Apply loose cropping with padding (k=0.20 = 20% padding)
        # This provides context around the face for better pose estimation
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)  # Less padding on bottom
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Load pose angles from .mat file (in radians)
        pose = get_ypr_from_mat(mat_path)
        # Convert to degrees for easier interpretation
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        
        # Angle binning: Convert continuous angles to discrete classes
        # 66 bins from -99° to +99° in 3° increments
        bins = np.array(range(-99, 102, 3))
        
        # Clip angles to valid range BEFORE binning to prevent out-of-bounds
        yaw_clipped = np.clip(yaw, -99, 99)
        pitch_clipped = np.clip(pitch, -99, 99)
        roll_clipped = np.clip(roll, -99, 99)
        
        # Digitize: Convert continuous values to bin indices
        binned_pose = np.digitize([yaw_clipped, pitch_clipped, roll_clipped], bins) - 1
        
        # Ensure valid range [0, 65] as final safety check
        binned_pose = np.clip(binned_pose, 0, 65)
        
        # Create tensors for both representations
        labels = torch.LongTensor(binned_pose)  # For classification loss
        cont_labels = torch.FloatTensor([yaw, pitch, roll])  # For regression loss & evaluation

        # Apply image transformations (resize, normalize, etc.)
        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        """Return dataset size"""
        # 2,000 samples in AFLW2000 dataset
        return self.length

class AFLW2000_Augmented(Dataset):
    """
    AFLW2000 Dataset with Data Augmentation for Training
    
    Extends the base AFLW2000 dataset with augmentation techniques:
    - Horizontal flipping (50% probability) with angle sign correction
    - Blur augmentation (5% probability) for robustness
    
    Augmentation Strategy:
    - Horizontal flip: yaw = -yaw, roll = -roll (pitch unchanged)
    - Random blur: Simulates motion blur or focus issues
    - These augmentations help the model generalize better
    """
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip augmentation (following original methodology)
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur augmentation (5% chance)
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        
        # Clip angles to valid range BEFORE binning
        yaw_clipped = np.clip(yaw, -99, 99)
        pitch_clipped = np.clip(pitch, -99, 99)
        roll_clipped = np.clip(roll, -99, 99)
        
        # Bin the clipped angles
        binned_pose = np.digitize([yaw_clipped, pitch_clipped, roll_clipped], bins) - 1
        
        # Ensure valid range [0, 65] as final safety check
        binned_pose = np.clip(binned_pose, 0, 65)
        
        # Create tensors
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])  # Keep original for evaluation

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        return self.length

def create_data_loaders(data_dir, splits_dir, batch_size=16, num_workers=0):
    """Create data loaders following original methodology"""
    
    print("=== Creating Data Loaders (Deep-Head-Pose Style) ===")
    
    # Exact transforms
    train_transforms = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Image Net normalization
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Image Net normalization
    ])
    
    # Create datasets
    train_dataset = AFLW2000_Augmented(
        data_dir=data_dir,
        filename_path=os.path.join(splits_dir, 'train.txt'),
        transform=train_transforms
    )
    
    val_dataset = AFLW2000(
        data_dir=data_dir, 
        filename_path=os.path.join(splits_dir, 'val.txt'),
        transform=test_transforms
    )
    
    test_dataset = AFLW2000(
        data_dir=data_dir,
        filename_path=os.path.join(splits_dir, 'test.txt'), 
        transform=test_transforms
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    print("Data loaders created successfully!")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loaders
    data_dir = "data/AFLW2000"
    splits_dir = "data/splits"
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, splits_dir, batch_size=4, num_workers=0
    )
    
    # Test one batch
    for images, labels, cont_labels, filenames in train_loader:
        print(f"Batch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels (binned): {labels.shape}")
        print(f"  Continuous labels: {cont_labels.shape}")
        print(f"  Sample angles (degrees): {cont_labels[0]}")
        print(f"  Sample filename: {filenames[0]}")
        break
    
    print("Dataset test completed successfully!")