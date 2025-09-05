# test_utils.py - Test utility functions
import sys
sys.path.append('.')
from utils.utils import get_ypr_from_mat, get_pt2d_from_mat
import numpy as np

def test_utils():
    """Test utility functions with your data"""
    
    mat_path = "data/AFLW2000/image00002.mat"
    
    print("=== Testing Utility Functions ===")
    
    # Test pose extraction
    pose = get_ypr_from_mat(mat_path)
    print(f"Pose (radians): {pose}")
    print(f"Pose (degrees): {np.degrees(pose)}")
    
    # Test landmarks extraction
    pt2d = get_pt2d_from_mat(mat_path)
    print(f"Landmarks shape: {pt2d.shape}")
    print(f"X range: {pt2d[0,:].min():.1f} to {pt2d[0,:].max():.1f}")
    print(f"Y range: {pt2d[1,:].min():.1f} to {pt2d[1,:].max():.1f}")
    
    print("Utility functions working correctly!")

if __name__ == "__main__":
    test_utils()