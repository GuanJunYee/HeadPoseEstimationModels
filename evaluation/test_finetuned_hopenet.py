# evaluation/test_finetuned_hopenet.py - Clear naming
import sys
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

sys.path.append('.')
from models.hopenet import create_hopenet
from utils.datasets import create_data_loaders
from training.finetune_hopenet import HopeNetFineTuner

def test_finetuned_hopenet_model():
    """Test the FINE-TUNED HopeNet model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data loaders
    print("Loading AFLW2000 data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        "data/AFLW2000", "data/splits", batch_size=16, num_workers=0
    )
    
    # Load the FINE-TUNED model (trained by fine-tuning their pre-trained model)
    print("Loading FINE-TUNED HopeNet model...")
    model = create_hopenet(num_bins=66)
    
    # Fix CUDA loading issue
    checkpoint = torch.load(
        "results/hopenet_finetuned/best_hopenet_finetuned.pth", 
        map_location=device  # This fixes the CUDA error
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get validation performance for comparison
    val_mae = checkpoint['best_val_mae']
    
    print("=" * 60)
    print("TESTING FINE-TUNED HOPENET MODEL")
    print("=" * 60)
    print(f"Model: Fine-tuned HopeNet (started from their pre-trained model)")
    print(f"Training method: Fine-tuning (20 epochs)")
    print(f"Validation MAE: {val_mae:.2f}°")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("-" * 60)
    
    # Create evaluator
    evaluator = HopeNetFineTuner(model, train_loader, val_loader, device)
    
    # Test on test set
    print("Running test evaluation...")
    model.eval()
    all_pred_angles = []
    all_true_angles = []
    
    with torch.no_grad():
        for images, labels, cont_labels, _ in test_loader:
            images = images.to(device)
            
            # Forward pass
            yaw_logits, pitch_logits, roll_logits = model(images)
            
            # Convert to angles using HopeNet method
            yaw_pred, pitch_pred, roll_pred = evaluator.convert_logits_to_angles(
                yaw_logits, pitch_logits, roll_logits
            )
            
            predicted_angles = torch.stack([yaw_pred, pitch_pred, roll_pred], dim=1)
            
            all_pred_angles.append(predicted_angles.cpu())
            all_true_angles.append(cont_labels)
    
    # Calculate final test performance
    all_pred_angles = torch.cat(all_pred_angles).numpy()
    all_true_angles = torch.cat(all_true_angles).numpy()
    
    overall_mae = mean_absolute_error(all_true_angles, all_pred_angles)
    yaw_mae = mean_absolute_error(all_true_angles[:, 0], all_pred_angles[:, 0])
    pitch_mae = mean_absolute_error(all_true_angles[:, 1], all_pred_angles[:, 1])
    roll_mae = mean_absolute_error(all_true_angles[:, 2], all_pred_angles[:, 2])
    
    print("=" * 60)
    print("FINAL TEST RESULTS - FINE-TUNED HOPENET")
    print("=" * 60)
    print(f"Overall Test MAE: {overall_mae:.2f}°")
    print(f"Yaw Test MAE: {yaw_mae:.2f}°")
    print(f"Pitch Test MAE: {pitch_mae:.2f}°")
    print(f"Roll Test MAE: {roll_mae:.2f}°")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Validation MAE: {val_mae:.2f}°")
    print(f"Test MAE: {overall_mae:.2f}°")
    
    difference = overall_mae - val_mae
    print(f"Difference: {difference:+.2f}°")
    
    if abs(difference) < 0.3:
        status = "✅ EXCELLENT! Very close performance (great generalization)"
    elif difference > 0.5:
        status = "⚠️ Test worse than validation (mild overfitting, still excellent)"
    elif difference < -0.3:
        status = "✅ Test better than validation (conservative validation set)"
    else:
        status = "✅ Good generalization"
    
    print(f"Status: {status}")
    
    print("\n" + "=" * 60)
    print("COMPLETE MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print("| Model                    | Method              | MAE     |")
    print("|--------------------------|---------------------|---------|")
    print("| Original Custom CNN      | From scratch        | 12.76°  |")
    print("| Custom CNN (Improved)    | HopeNet methodology | 9.64°   |")
    print("| HopeNet (From scratch)   | Deep-Head-Pose      | 7.31°   |")
    print(f"| HopeNet (Fine-tuned)     | Fine-tuning         | {overall_mae:.2f}°   |")
    print("=" * 60)
    
    # Calculate improvements
    original_mae = 12.76
    improvement = ((original_mae - overall_mae) / original_mae) * 100
    
    print(f"\nFINAL IMPROVEMENT: {improvement:.1f}% better than original Custom CNN!")
    
    return {
        'test_mae': overall_mae,
        'test_yaw_mae': yaw_mae,
        'test_pitch_mae': pitch_mae,
        'test_roll_mae': roll_mae,
        'val_mae': val_mae,
        'improvement_vs_original': improvement
    }

if __name__ == "__main__":
    results = test_finetuned_hopenet_model()