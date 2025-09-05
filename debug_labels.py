# debug_labels.py - Check what's happening with the labels
import sys
sys.path.append('.')
from utils.datasets import create_data_loaders
import numpy as np

def debug_labels():
    """Debug the label binning issue"""
    
    data_dir = "data/AFLW2000"
    splits_dir = "data/splits"
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, splits_dir, batch_size=4, num_workers=0
    )
    
    print("=== Debugging Label Ranges ===")
    
    all_cont_labels = []
    all_binned_labels = []
    
    # Check first few batches
    for batch_idx, (images, labels, cont_labels, filenames) in enumerate(train_loader):
        all_cont_labels.append(cont_labels.numpy())
        all_binned_labels.append(labels.numpy())
        
        if batch_idx >= 10:  # Check first 10 batches
            break
    
    all_cont_labels = np.concatenate(all_cont_labels)
    all_binned_labels = np.concatenate(all_binned_labels)
    
    print(f"Continuous labels shape: {all_cont_labels.shape}")
    print(f"Binned labels shape: {all_binned_labels.shape}")
    
    print(f"\nContinuous label ranges:")
    print(f"  Yaw: {all_cont_labels[:, 0].min():.1f} to {all_cont_labels[:, 0].max():.1f}")
    print(f"  Pitch: {all_cont_labels[:, 1].min():.1f} to {all_cont_labels[:, 1].max():.1f}")
    print(f"  Roll: {all_cont_labels[:, 2].min():.1f} to {all_cont_labels[:, 2].max():.1f}")
    
    print(f"\nBinned label ranges:")
    print(f"  Yaw: {all_binned_labels[:, 0].min()} to {all_binned_labels[:, 0].max()}")
    print(f"  Pitch: {all_binned_labels[:, 1].min()} to {all_binned_labels[:, 1].max()}")
    print(f"  Roll: {all_binned_labels[:, 2].min()} to {all_binned_labels[:, 2].max()}")
    
    # Check for invalid indices
    invalid_yaw = (all_binned_labels[:, 0] < 0) | (all_binned_labels[:, 0] >= 66)
    invalid_pitch = (all_binned_labels[:, 1] < 0) | (all_binned_labels[:, 1] >= 66)
    invalid_roll = (all_binned_labels[:, 2] < 0) | (all_binned_labels[:, 2] >= 66)
    
    print(f"\nInvalid indices:")
    print(f"  Invalid yaw indices: {invalid_yaw.sum()}")
    print(f"  Invalid pitch indices: {invalid_pitch.sum()}")
    print(f"  Invalid roll indices: {invalid_roll.sum()}")
    
    if invalid_yaw.sum() > 0:
        print(f"  Problem yaw values: {all_cont_labels[invalid_yaw, 0]}")
    if invalid_pitch.sum() > 0:
        print(f"  Problem pitch values: {all_cont_labels[invalid_pitch, 1]}")
    if invalid_roll.sum() > 0:
        print(f"  Problem roll values: {all_cont_labels[invalid_roll, 2]}")

if __name__ == "__main__":
    debug_labels()