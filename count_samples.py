# count_samples.py - Count total AFLW2000 samples
import os
import glob

def count_aflw2000_samples():
    """Count total samples in AFLW2000"""
    
    data_dir = "data/AFLW2000"
    
    # Count .jpg files
    jpg_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    
    print(f"=== AFLW2000 Sample Count ===")
    print(f"Image files (.jpg): {len(jpg_files)}")
    print(f"Annotation files (.mat): {len(mat_files)}")
    
    if len(jpg_files) == len(mat_files):
        print(f"Perfect match! {len(jpg_files)} complete samples")
        return len(jpg_files)
    else:
        print(f"Mismatch! Missing some .jpg or .mat files")
        return 0

if __name__ == "__main__":
    count_aflw2000_samples()