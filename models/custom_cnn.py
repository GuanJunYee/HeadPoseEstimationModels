# models/custom_cnn.py
"""
Custom CNN Implementation for Head Pose Estimation

This module implements a custom CNN architecture that adapts HopeNet's dual-loss methodology
to a lightweight, from-scratch design.

ARCHITECTURE PHILOSOPHY:
Unlike HopeNet's ResNet-50 backbone, this custom CNN is designed with:
- Lightweight architecture for faster training and inference
- Progressive feature extraction through 4 convolutional blocks
- Batch normalization and dropout for regularization
- Three separate classification heads following HopeNet's multi-head design

METHODOLOGY ADAPTATION:
This model adopts HopeNet's key innovations:
1. Dual-loss function (classification + regression)
2. Angle binning (66 bins from -99° to +99°)
3. Expected value calculation for continuous angles
4. Multi-head architecture for yaw, pitch, roll

ARCHITECTURE DETAILS:
- 4 convolutional blocks with progressive channel expansion (32→64→128→256)
- Batch normalization after each convolution for training stability
- MaxPooling for spatial dimension reduction
- Global average pooling before classification heads
- Dropout (0.5) for regularization
- Three separate FC heads for angle prediction

TRAINING ADVANTAGES:
- Fewer parameters than ResNet-50 (faster training)
- End-to-end learnable (no pre-trained backbone dependency)
- Same dual-loss methodology as HopeNet for comparable performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomHeadPoseCNN(nn.Module):
    """
    Custom CNN for Head Pose Estimation using HopeNet Methodology
    
    This lightweight CNN architecture adopts HopeNet's dual-loss approach while
    using a simpler backbone design for efficient training and inference.
    
    Architecture Flow:
    Input (224x224x3) → Conv Blocks → Global Pool → 3 FC Heads → Angle Predictions
    
    Key Features:
    - Progressive feature extraction: 32→64→128→256 channels
    - Batch normalization for training stability
    - Dropout regularization to prevent overfitting
    - Three separate heads for yaw, pitch, roll (following HopeNet)
    - Same 66-bin classification approach as HopeNet
    """
    
    def __init__(self, num_bins=66):
        super(CustomHeadPoseCNN, self).__init__()
        
        # Your original architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        
        # Calculate the size after convolutions and pooling
        # Input: 224x224 -> After 4 pools: 14x14
        self.fc_input_size = 256 * 14 * 14
        
        # Feature extraction layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # NEW: Three classification heads (following HopeNet approach)
        self.fc_yaw = nn.Linear(256, num_bins)
        self.fc_pitch = nn.Linear(256, num_bins)
        self.fc_roll = nn.Linear(256, num_bins)

        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """
        Forward pass through Custom CNN
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            Tuple of (yaw_logits, pitch_logits, roll_logits)
            Each tensor has shape [batch_size, num_bins]
            
        Architecture Flow:
        1. Progressive feature extraction through 4 conv blocks
        2. Spatial dimension reduction via max pooling
        3. Feature refinement through FC layers  
        4. Three separate classification heads for angles
        
        Spatial Dimension Changes:
        224x224 → 112x112 → 56x56 → 28x28 → 14x14 (after 4 pooling ops)
        """
        # Progressive feature extraction with pooling after each block
        # Conv Block 1: [B,3,224,224] → [B,32,112,112]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2: [B,32,112,112] → [B,64,56,56]  
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3: [B,64,56,56] → [B,128,28,28]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4: [B,128,28,28] → [B,256,14,14]
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten spatial dimensions: [B,256,14,14] → [B,256*14*14]
        x = x.view(x.size(0), -1)
        
        # Feature refinement through fully connected layers
        x = self.relu(self.fc1(x))  # [B,50176] → [B,512]
        x = self.dropout(x)         # Regularization
        x = self.relu(self.fc2(x))  # [B,512] → [B,256]
        x = self.dropout(x)         # Regularization
        
        # Three separate classification heads (following HopeNet methodology)
        # Each head predicts probability distribution over angle bins
        yaw_logits = self.fc_yaw(x)     # [B,256] → [B,66] - yaw angle bins
        pitch_logits = self.fc_pitch(x) # [B,256] → [B,66] - pitch angle bins
        roll_logits = self.fc_roll(x)   # [B,256] → [B,66] - roll angle bins
        
        return yaw_logits, pitch_logits, roll_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_custom_cnn(num_bins=66):
    """Create improved Custom CNN model"""  
    model = CustomHeadPoseCNN(num_bins)
    return model

if __name__ == "__main__":
    # Test the improved model
    model = create_custom_cnn(num_bins=66)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"Improved Custom CNN parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    yaw, pitch, roll = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Yaw output shape: {yaw.shape}")
    print(f"Pitch output shape: {pitch.shape}")
    print(f"Roll output shape: {roll.shape}")
    
    print("Improved Custom CNN model test completed successfully!")