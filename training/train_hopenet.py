# training/train_hopenet.py (Updated)
"""
HopeNet Training Implementation

This module implements the training procedure for HopeNet following the dual-loss methodology
from "Fine-Grained Head Pose Estimation Without Keypoints" by Ruiz et al.

TRAINING METHODOLOGY:
HopeNet uses a novel dual-loss approach that combines:

1. CLASSIFICATION LOSS:
   - Treats angle prediction as a classification problem
   - 66 bins from -99° to +99° in 3° increments
   - Uses CrossEntropyLoss on binned angle labels
   - Helps with discrete angle learning

2. REGRESSION LOSS:
   - Predicts continuous angles using expected value calculation
   - Converts softmax probabilities to continuous angles
   - Uses MSELoss on continuous angle predictions
   - Helps with fine-grained angle precision

3. COMBINED LOSS:
   Total Loss = Classification Loss + α × Regression Loss
   Where α = 0.001 (balances the two loss components)

KEY INNOVATIONS:
- Angle binning converts regression to classification
- Expected value calculation from probability distributions
- Multi-head architecture for yaw, pitch, roll
- Different learning rates for backbone vs heads
- Early stopping with validation monitoring

TRAINING FEATURES:
- ResNet-50 backbone (pre-trained on ImageNet)
- Different learning rates for backbone (0.1×) vs heads (1.0×)
- Weight decay regularization (1e-4)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for training stability
- Early stopping based on validation loss
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

sys.path.append('.')
from models.hopenet import create_hopenet
from utils.datasets import create_data_loaders

class HopeNetTrainer:
    """
    HopeNet Training Class
    
    Implements the complete training pipeline for HopeNet including:
    - Dual-loss function (classification + regression)
    - Multi-head optimization with different learning rates
    - Early stopping and model checkpointing
    - Training progress visualization
    
    The trainer follows the exact methodology from the original paper
    to ensure reproducible results.
    """
    
    def __init__(self, model, train_loader, val_loader, device, save_dir="results/hopenet"):
        """
        Initialize HopeNet trainer
        
        Args:
            model: HopeNet model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device (cuda/cpu)
            save_dir: Directory to save results and checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss functions for dual-loss methodology
        self.criterion = nn.CrossEntropyLoss()  # Classification loss for binned angles
        self.reg_criterion = nn.MSELoss()       # Regression loss for continuous angles
        self.alpha = 0.001  # Regression loss coefficient (balances cls vs reg)
        
        # Angle conversion: bin indices to continuous angles
        # Creates tensor [0, 1, 2, ..., 65] for expected value calculation
        self.idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(device)
        
        # Training history tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_yaw_mae': [],
            'val_pitch_mae': [],
            'val_roll_mae': [],
            'learning_rates': []
        }
        
        self.best_val_mae = float('inf')
        self.best_val_loss = float('inf')
    
    def setup_optimizer(self, lr=0.001):
        """Setup optimizer with regularization"""
        # Different learning rates for backbone vs heads
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if any(head_name in name for head_name in ['fc_yaw', 'fc_pitch', 'fc_roll']):
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': lr, 'weight_decay': 1e-4}
        ], lr=lr)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print(f"HopeNet Training Setup:")
        print(f"  Backbone LR: {lr * 0.1} (with weight decay)")
        print(f"  Heads LR: {lr} (with weight decay)")
        print(f"  LR Scheduler: ReduceLROnPlateau")
    
    def convert_logits_to_angles(self, yaw_logits, pitch_logits, roll_logits):
        """
        Convert classification logits to continuous angles using expected value
        
        This is the core innovation of HopeNet: converting discrete predictions
        back to continuous values using probability distributions.
        
        Args:
            yaw_logits, pitch_logits, roll_logits: Raw model outputs [batch_size, 66]
            
        Returns:
            Continuous angle predictions in degrees
            
        Process:
        1. Apply softmax to get probability distributions
        2. Calculate expected value: Σ(probability × bin_center)
        3. Convert bin indices to actual angles: index * 3 - 99
        
        Mathematical Formula:
        angle = Σ(i=0 to 65) P(bin_i) × (i × 3 - 99)
        Where P(bin_i) is the softmax probability for bin i
        """
        # Convert logits to probability distributions
        yaw_probs = F.softmax(yaw_logits, dim=1)     # [batch_size, 66]
        pitch_probs = F.softmax(pitch_logits, dim=1) # [batch_size, 66]
        roll_probs = F.softmax(roll_logits, dim=1)   # [batch_size, 66]
        
        # Calculate expected value (weighted sum of bin centers)
        # idx_tensor = [0, 1, 2, ..., 65]
        # Angle = (weighted_sum * 3) - 99 converts indices to degrees
        yaw_pred = torch.sum(yaw_probs * self.idx_tensor, 1) * 3 - 99
        pitch_pred = torch.sum(pitch_probs * self.idx_tensor, 1) * 3 - 99
        roll_pred = torch.sum(roll_probs * self.idx_tensor, 1) * 3 - 99
        
        return yaw_pred, pitch_pred, roll_pred
    
    def calculate_loss(self, yaw_logits, pitch_logits, roll_logits,
                      yaw_labels, pitch_labels, roll_labels,
                      yaw_cont, pitch_cont, roll_cont):
        """
        Calculate HopeNet's dual loss function
        
        The dual loss combines classification and regression objectives:
        1. Classification Loss: CrossEntropy on binned angle labels
        2. Regression Loss: MSE on continuous angle predictions
        
        Args:
            *_logits: Raw model outputs [batch_size, 66]
            *_labels: Binned angle labels [batch_size] for classification
            *_cont: Continuous angle labels [batch_size] for regression
            
        Returns:
            Dictionary containing all loss components and predictions
            
        Loss Formulation:
        Total Loss = Classification Loss + α × Regression Loss
        
        Where:
        - Classification Loss encourages learning discrete angle bins
        - Regression Loss ensures continuous angle precision
        - α = 0.001 balances the two components
        """
        # Classification losses for each angle (binned predictions)
        loss_yaw_cls = self.criterion(yaw_logits, yaw_labels)
        loss_pitch_cls = self.criterion(pitch_logits, pitch_labels)
        loss_roll_cls = self.criterion(roll_logits, roll_labels)
        
        # Convert logits to continuous angles using expected value
        yaw_pred, pitch_pred, roll_pred = self.convert_logits_to_angles(
            yaw_logits, pitch_logits, roll_logits
        )
        
        # Regression losses for continuous angle prediction
        loss_yaw_reg = self.reg_criterion(yaw_pred, yaw_cont)
        loss_pitch_reg = self.reg_criterion(pitch_pred, pitch_cont)
        loss_roll_reg = self.reg_criterion(roll_pred, roll_cont)
        
        # Combined losses (classification + α × regression)
        loss_yaw = loss_yaw_cls + self.alpha * loss_yaw_reg
        loss_pitch = loss_pitch_cls + self.alpha * loss_pitch_reg
        loss_roll = loss_roll_cls + self.alpha * loss_roll_reg
        
        # Total loss across all three angles
        total_loss = loss_yaw + loss_pitch + loss_roll
        
        # Stack predictions for evaluation
        predicted_angles = torch.stack([yaw_pred, pitch_pred, roll_pred], dim=1)
        
        return {
            'total_loss': total_loss,
            'yaw_loss': loss_yaw,
            'pitch_loss': loss_pitch,
            'roll_loss': loss_roll,
            'predicted_angles': predicted_angles
        }
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, labels, cont_labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            yaw_labels = labels[:, 0].to(self.device)
            pitch_labels = labels[:, 1].to(self.device)
            roll_labels = labels[:, 2].to(self.device)
            
            yaw_cont = cont_labels[:, 0].to(self.device)
            pitch_cont = cont_labels[:, 1].to(self.device)
            roll_cont = cont_labels[:, 2].to(self.device)
            
            yaw_logits, pitch_logits, roll_logits = self.model(images)
            
            loss_dict = self.calculate_loss(
                yaw_logits, pitch_logits, roll_logits,
                yaw_labels, pitch_labels, roll_labels,
                yaw_cont, pitch_cont, roll_cont
            )
            
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'    Batch [{batch_idx+1}/{len(self.train_loader)}] '
                      f'Losses: Yaw {loss_dict["yaw_loss"].item():.4f}, '
                      f'Pitch {loss_dict["pitch_loss"].item():.4f}, '
                      f'Roll {loss_dict["roll_loss"].item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate epoch"""
        self.model.eval()
        total_loss = 0
        all_pred_angles = []
        all_true_angles = []
        
        with torch.no_grad():
            for images, labels, cont_labels, _ in self.val_loader:
                images = images.to(self.device)
                
                yaw_labels = labels[:, 0].to(self.device)
                pitch_labels = labels[:, 1].to(self.device)
                roll_labels = labels[:, 2].to(self.device)
                
                yaw_cont = cont_labels[:, 0].to(self.device)
                pitch_cont = cont_labels[:, 1].to(self.device)
                roll_cont = cont_labels[:, 2].to(self.device)
                
                yaw_logits, pitch_logits, roll_logits = self.model(images)
                
                loss_dict = self.calculate_loss(
                    yaw_logits, pitch_logits, roll_logits,
                    yaw_labels, pitch_labels, roll_labels,
                    yaw_cont, pitch_cont, roll_cont
                )
                
                total_loss += loss_dict['total_loss'].item()
                all_pred_angles.append(loss_dict['predicted_angles'].cpu())
                all_true_angles.append(cont_labels)
        
        all_pred_angles = torch.cat(all_pred_angles).numpy()
        all_true_angles = torch.cat(all_true_angles).numpy()
        
        overall_mae = mean_absolute_error(all_true_angles, all_pred_angles)
        yaw_mae = mean_absolute_error(all_true_angles[:, 0], all_pred_angles[:, 0])
        pitch_mae = mean_absolute_error(all_true_angles[:, 1], all_pred_angles[:, 1])
        roll_mae = mean_absolute_error(all_true_angles[:, 2], all_pred_angles[:, 2])
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_mae': overall_mae,
            'yaw_mae': yaw_mae,
            'pitch_mae': pitch_mae,
            'roll_mae': roll_mae
        }
    
    def train(self, num_epochs=100, early_stopping_patience=20):
        """Training with early stopping"""
        print("\n=== HopeNet From Scratch Training with Early Stopping ===")
        
        self.setup_optimizer(lr=0.001)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_mae'].append(val_metrics['val_mae'])
            self.history['val_yaw_mae'].append(val_metrics['yaw_mae'])
            self.history['val_pitch_mae'].append(val_metrics['pitch_mae'])
            self.history['val_roll_mae'].append(val_metrics['roll_mae'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Step scheduler
            self.scheduler.step(val_metrics['val_loss'])
            
            # Check for best model
            if val_metrics['val_mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['val_mae']
                self.save_best_model(epoch + 1, val_metrics)
                best_marker = " ⭐ BEST MAE!"
            else:
                best_marker = ""
            
            # Early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val MAE: {val_metrics['val_mae']:.2f}° "
                  f"(Yaw: {val_metrics['yaw_mae']:.2f}°, "
                  f"Pitch: {val_metrics['pitch_mae']:.2f}°, "
                  f"Roll: {val_metrics['roll_mae']:.2f}°){best_marker}")
            print(f"  Current LR: {current_lr:.2e}, Early Stop Counter: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\n=== HopeNet From Scratch Training Complete ===")
        print(f"Best Val MAE: {self.best_val_mae:.2f}°")
        
        self.create_plots()
    
    def save_best_model(self, epoch, metrics):
        """Save best model"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_mae': self.best_val_mae,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, 'best_hopenet.pth'))
    
    def create_plots(self):
        """Create training plots"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE plot
        ax1.plot(epochs, self.history['val_mae'], 'r-', label='Overall MAE')
        ax1.plot(epochs, self.history['val_yaw_mae'], 'b--', label='Yaw MAE')
        ax1.plot(epochs, self.history['val_pitch_mae'], 'g--', label='Pitch MAE')
        ax1.plot(epochs, self.history['val_roll_mae'], 'm--', label='Roll MAE')
        ax1.set_title('HopeNet From Scratch - Mean Absolute Error')
        ax1.set_ylabel('MAE (degrees)')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax2.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax2.set_title('HopeNet From Scratch - Loss Progress')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Final MAE by angle
        ax3.bar(['Yaw', 'Pitch', 'Roll'], 
                [self.history['val_yaw_mae'][-1], 
                self.history['val_pitch_mae'][-1], 
                self.history['val_roll_mae'][-1]])
        ax3.set_title('HopeNet From Scratch - Final MAE by Angle')
        ax3.set_ylabel('MAE (degrees)')
        ax3.grid(True)
        
        # Validation MAE progress
        ax4.plot(epochs, self.history['val_mae'], 'r-')
        ax4.axhline(y=self.best_val_mae, color='g', linestyle='--', 
                    label=f'Best MAE: {self.best_val_mae:.2f}°')
        ax4.set_title('HopeNet From Scratch - Validation MAE Progress')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('MAE (degrees)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'hopenet_training_curves.png'), dpi=300)
        plt.show()

def main():
   """Main training function"""
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   data_dir = "data/AFLW2000"
   splits_dir = "data/splits"
   
   train_loader, val_loader, test_loader = create_data_loaders(
       data_dir, splits_dir, batch_size=16, num_workers=0
   )
   
   model = create_hopenet(num_bins=66)
   
   print(f"HopeNet Model Info:")
   print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
   
   trainer = HopeNetTrainer(model, train_loader, val_loader, device)
   trainer.train(num_epochs=100, early_stopping_patience=20)

if __name__ == "__main__":
   main()