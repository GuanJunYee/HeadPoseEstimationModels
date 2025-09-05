# create_data_splits.py - Create train/val/test splits
import os
import random
import glob

def create_aflw2000_splits():
    """Create train/val/test splits for AFLW2000"""
    
    data_dir = "data/AFLW2000"
    splits_dir = "data/splits"
    
    # Create splits directory
    os.makedirs(splits_dir, exist_ok=True)
    
    # Get all image files
    jpg_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    # Extract filename without extension and path
    filenames = []
    for jpg_file in jpg_files:
        basename = os.path.basename(jpg_file)
        filename_no_ext = os.path.splitext(basename)[0]
        filenames.append(filename_no_ext)
    
    # Sort for reproducibility
    filenames.sort()
    
    # Set random seed for reproducible splits
    random.seed(42)
    random.shuffle(filenames)
    
    total_samples = len(filenames)
    train_size = int(0.7 * total_samples)  # 70% train
    val_size = int(0.15 * total_samples)   # 15% validation
    # Remaining 15% for test
    
    train_files = filenames[:train_size]
    val_files = filenames[train_size:train_size + val_size]
    test_files = filenames[train_size + val_size:]
    
    print(f"=== AFLW2000 Data Splits ===")
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(train_files)} samples ({len(train_files)/total_samples*100:.1f}%)")
    print(f"Val: {len(val_files)} samples ({len(val_files)/total_samples*100:.1f}%)")
    print(f"Test: {len(test_files)} samples ({len(test_files)/total_samples*100:.1f}%)")
    
    # Save splits to text files
    def save_split(filenames, split_name):
        filepath = os.path.join(splits_dir, f"{split_name}.txt")
        with open(filepath, 'w') as f:
            for filename in filenames:
                f.write(f"{filename}\n")
        print(f"Saved {split_name}.txt with {len(filenames)} files")
    
    save_split(train_files, "train")
    save_split(val_files, "val")
    save_split(test_files, "test")
    
    print("\n=== Data splits created successfully! ===")

if __name__ == "__main__":
    create_aflw2000_splits()