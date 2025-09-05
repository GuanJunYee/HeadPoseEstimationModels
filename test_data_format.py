# test_data_format.py - Quick verification script
import scipy.io as sio
import numpy as np
import cv2
import os

def test_aflw2000_format():
    """Test if AFLW2000 data has correct format"""
    
    data_dir = "data/AFLW2000"
    
    # Test one sample
    test_image = "image00002.jpg"
    test_mat = "image00002.mat"
    
    image_path = os.path.join(data_dir, test_image)
    mat_path = os.path.join(data_dir, test_mat)
    
    print("=== Testing AFLW2000 Data Format ===")
    
    # Test image loading
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        print(f"Image loaded: {img.shape}")
    else:
        print(f"Image not found: {image_path}")
        return
    
    # Test .mat file loading
    if os.path.exists(mat_path):
        try:
            mat = sio.loadmat(mat_path)
            print(f"Mat file loaded. Keys: {list(mat.keys())}")
            
            # Check for required data
            if 'Pose_Para' in mat:
                pose_para = mat['Pose_Para'][0]
                print(f"Pose_Para found: {pose_para[:3]} (first 3: pitch, yaw, roll)")
                print(f"Range: pitch={pose_para[0]:.3f}, yaw={pose_para[1]:.3f}, roll={pose_para[2]:.3f}")
            else:
                print(f"Pose_Para not found in mat file")
                
            if 'pt2d' in mat:
                pt2d = mat['pt2d']
                print(f"pt2d landmarks found: {pt2d.shape}")
                print(f"X range: {pt2d[0,:].min():.1f} to {pt2d[0,:].max():.1f}")
                print(f"Y range: {pt2d[1,:].min():.1f} to {pt2d[1,:].max():.1f}")
            else:
                print(f"pt2d landmarks not found in mat file")
                
        except Exception as e:
            print(f"Error loading mat file: {e}")
    else:
        print(f"Mat file not found: {mat_path}")
    
    print("\n=== Data Format Check Complete ===")

if __name__ == "__main__":
    test_aflw2000_format()