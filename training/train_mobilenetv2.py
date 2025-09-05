# training/train_mobilenetv2.py
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('.')
from models.mobilenetv2 import create_mobilenetv2_headpose
from utils.datasets import create_data_loaders

class MobileNetV2:
    """Regularized MobileNetV2 trainer to fix validation loss divergence"""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir="results/mobilenetv2"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Same loss setup as HopeNet
        self.criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        self.alpha = 0.001  # Regression loss weight
        
        # Angle conversion setup
        self.idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(device)
        
        # Training history
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
        """Setup optimizer with regularization and learning rate scheduling"""
        
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if any(head_name in name for head_name in ['fc_yaw', 'fc_pitch', 'fc_roll']):
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # Reduced learning rates and weight decay for regularization
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.05, 'weight_decay': 1e-4},  # Much lower backbone LR
            {'params': head_params, 'lr': lr * 0.3, 'weight_decay': 1e-4}        # Lower head LR + weight decay
        ], lr=lr)
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',           # Minimize validation loss
            factor=0.5,          # Reduce LR by half
            patience=10,         # Wait 10 epochs before reducing
            threshold=0.01,      # Minimum change to qualify as improvement
            min_lr=1e-6,         # Minimum learning rate
            verbose=True         # Print when LR is reduced
        )
        
        print(f"Regularized Training Setup:")
        print(f"  Backbone LR: {lr * 0.05:.6f} (with weight decay 1e-4)")
        print(f"  Heads LR: {lr * 0.3:.6f} (with weight decay 1e-4)")
        print(f"  Dropout rate: 0.5")
        print(f"  LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
        print(f"  Total trainable parameters: {self.model.count_parameters():,}")
    
    def convert_logits_to_angles(self, yaw_logits, pitch_logits, roll_logits):
        """Convert logits to angles using HopeNet's method"""
        
        yaw_probs = F.softmax(yaw_logits, dim=1)
        pitch_probs = F.softmax(pitch_logits, dim=1)
        roll_probs = F.softmax(roll_logits, dim=1)
        
        yaw_pred = torch.sum(yaw_probs * self.idx_tensor, 1) * 3 - 99
        pitch_pred = torch.sum(pitch_probs * self.idx_tensor, 1) * 3 - 99
        roll_pred = torch.sum(roll_probs * self.idx_tensor, 1) * 3 - 99
        
        return yaw_pred, pitch_pred, roll_pred
    
    def calculate_loss(self, yaw_logits, pitch_logits, roll_logits,
                      yaw_labels, pitch_labels, roll_labels,
                      yaw_cont, pitch_cont, roll_cont):
        """Calculate loss using HopeNet's dual-loss approach"""
        
        # Classification losses
        loss_yaw_cls = self.criterion(yaw_logits, yaw_labels)
        loss_pitch_cls = self.criterion(pitch_logits, pitch_labels)
        loss_roll_cls = self.criterion(roll_logits, roll_labels)
        
        # Convert to continuous angles
        yaw_pred, pitch_pred, roll_pred = self.convert_logits_to_angles(
            yaw_logits, pitch_logits, roll_logits
        )
        
        # Regression losses
        loss_yaw_reg = self.reg_criterion(yaw_pred, yaw_cont)
        loss_pitch_reg = self.reg_criterion(pitch_pred, pitch_cont)
        loss_roll_reg = self.reg_criterion(roll_pred, roll_cont)
        
        # Combined losses
        loss_yaw = loss_yaw_cls + self.alpha * loss_yaw_reg
        loss_pitch = loss_pitch_cls + self.alpha * loss_pitch_reg
        loss_roll = loss_roll_cls + self.alpha * loss_roll_reg
        
        total_loss = loss_yaw + loss_pitch + loss_roll
        
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
        self.model.train()  # Enable dropout during training
        total_loss = 0
        
        for batch_idx, (images, labels, cont_labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # Labels
            yaw_labels = labels[:, 0].to(self.device)
            pitch_labels = labels[:, 1].to(self.device)
            roll_labels = labels[:, 2].to(self.device)
            
            yaw_cont = cont_labels[:, 0].to(self.device)
            pitch_cont = cont_labels[:, 1].to(self.device)
            roll_cont = cont_labels[:, 2].to(self.device)
            
            # Forward pass
            yaw_logits, pitch_logits, roll_logits = self.model(images)
            
            # Calculate loss
            loss_dict = self.calculate_loss(
                yaw_logits, pitch_logits, roll_logits,
                yaw_labels, pitch_labels, roll_labels,
                yaw_cont, pitch_cont, roll_cont
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f'    Batch [{batch_idx+1}/{len(self.train_loader)}] '
                      f'Losses: Yaw {loss_dict["yaw_loss"].item():.4f}, '
                      f'Pitch {loss_dict["pitch_loss"].item():.4f}, '
                      f'Roll {loss_dict["roll_loss"].item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate epoch"""
        self.model.eval()  # Disable dropout during validation
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
        """Training with early stopping and regularization"""
        print("\n=== MobileNetV2 Regularized Training ===")
        
        # Setup optimizer with regularization
        self.setup_optimizer(lr=0.001)
        
        # Early stopping variables
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_mae'].append(val_metrics['val_mae'])
            self.history['val_yaw_mae'].append(val_metrics['yaw_mae'])
            self.history['val_pitch_mae'].append(val_metrics['pitch_mae'])
            self.history['val_roll_mae'].append(val_metrics['roll_mae'])
            
            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Step learning rate scheduler
            self.scheduler.step(val_metrics['val_loss'])
            
            # Check for best model (based on MAE)
            if val_metrics['val_mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['val_mae']
                self.save_best_model(epoch + 1, val_metrics)
                best_marker = " ⭐ BEST MAE!"
            else:
                best_marker = ""
            
            # Early stopping based on validation loss
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print results
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val MAE: {val_metrics['val_mae']:.2f}° "
                  f"(Yaw: {val_metrics['yaw_mae']:.2f}°, "
                  f"Pitch: {val_metrics['pitch_mae']:.2f}°, "
                  f"Roll: {val_metrics['roll_mae']:.2f}°){best_marker}")
            print(f"  Current LR: {current_lr:.2e}, Early Stop Counter: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered after {early_stopping_patience} epochs without validation loss improvement")
                print(f"Training stopped at epoch {epoch+1}")
                break
        
        print(f"\n=== MobileNetV2 Regularized Training Complete ===")
        print(f"Best Val MAE: {self.best_val_mae:.2f}°")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # Create plots
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
        torch.save(checkpoint, os.path.join(self.save_dir, 'best_mobilenetv2.pth'))
    
    def create_plots(self):
        """Create training plots"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE plot
        ax1.plot(epochs, self.history['val_mae'], 'r-', label='Overall MAE')
        ax1.plot(epochs, self.history['val_yaw_mae'], 'b--', label='Yaw MAE')
        ax1.plot(epochs, self.history['val_pitch_mae'], 'g--', label='Pitch MAE')
        ax1.plot(epochs, self.history['val_roll_mae'], 'm--', label='Roll MAE')
        ax1.set_title('MobileNetV2 - Mean Absolute Error')
        ax1.set_ylabel('MAE (degrees)')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot (SHOULD BE FIXED NOW!)
        ax2.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax2.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax2.set_title('MobileNetV2 - Loss Progress')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # MAE comparison
        ax3.bar(['Yaw', 'Pitch', 'Roll'], 
               [self.history['val_yaw_mae'][-1], 
                self.history['val_pitch_mae'][-1], 
                self.history['val_roll_mae'][-1]])
        ax3.set_title('MobileNetV2 - Final MAE by Angle')
        ax3.set_ylabel('MAE (degrees)')
        ax3.grid(True)
        
        # Learning rate plot
        ax4.plot(epochs, self.history['val_mae'], 'r-')
        ax4.axhline(y=self.best_val_mae, color='g', linestyle='--', 
                label=f'Best MAE: {self.best_val_mae:.2f}°')
        ax4.set_title('MobileNetV2 - Validation MAE Progress')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('MAE (degrees)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'mobilenetv2_training_curves.png'), dpi=300)
        plt.show()

def main():
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    data_dir = "data/AFLW2000"
    splits_dir = "data/splits"
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, splits_dir, batch_size=16, num_workers=0
    )
    
    # Create regularized MobileNetV2 model
    model = create_mobilenetv2_headpose(num_bins=66, pretrained=True, dropout_rate=0.5)
    
    print(f"MobileNetV2 Regularized Model Info:")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")
    print(f"  Dropout rate: 0.5")
    print(f"  Regularization: Weight decay + LR scheduling + Early stopping")
    
    # Create trainer
    trainer = MobileNetV2(model, train_loader, val_loader, device)
    
    # Train the model with regularization
    trainer.train(num_epochs=100, early_stopping_patience=20)

if __name__ == "__main__":
    main()