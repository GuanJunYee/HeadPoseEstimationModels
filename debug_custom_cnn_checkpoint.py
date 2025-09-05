# debug_custom_cnn_checkpoint.py
import torch

def check_checkpoint():
    """Check what's in the Custom CNN checkpoint"""
    try:
        checkpoint = torch.load("results/custom_cnn/best_custom_cnn.pth", map_location='cpu')
        
        print("=== Custom CNN Checkpoint Info ===")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best Val MAE: {checkpoint.get('best_val_mae', 'Unknown')}")
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            print(f"Training epochs completed: {len(history.get('val_mae', []))}")
            if history.get('val_mae'):
                print(f"Final validation MAE: {history['val_mae'][-1]:.2f}°")
                print(f"Best validation MAE in history: {min(history['val_mae']):.2f}°")
        
        # Check if this matches your expected 9.64°
        expected_mae = 9.64
        actual_mae = checkpoint.get('best_val_mae', float('inf'))
        
        if abs(actual_mae - expected_mae) < 0.5:
            print("This looks like the correct model (close to 9.64°)")
        else:
            print(f"This doesn't match expected 9.64° (got {actual_mae}°)")
            print("The checkpoint might be from a different training run")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    check_checkpoint()