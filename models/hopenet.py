# models/hopenet.py
"""
HopeNet (Head Orientation Prediction Network) Implementation

This is a PyTorch implementation of HopeNet, a deep learning model for 3D head pose estimation.
The paper: "Fine-Grained Head Pose Estimation Without Keypoints" by Ruiz et al.

ARCHITECTURE OVERVIEW:
HopeNet uses a dual-loss approach that combines:
1. Classification: Bins continuous angles into discrete classes (66 bins from -99° to +99°)
2. Regression: Predicts expected value from the probability distribution

KEY INNOVATIONS:
- Dual-loss function combining classification and regression losses
- Angle binning: Converts continuous angle prediction into classification problem
- Multi-head output: Separate prediction heads for yaw, pitch, and roll
- Expected value calculation: Uses softmax probabilities to compute continuous angles

METHODOLOGY:
- Uses ResNet-50 backbone (pre-trained on ImageNet) for feature extraction
- Three separate fully connected heads for yaw, pitch, roll prediction
- Each head outputs 66-dimensional probability distribution over angle bins
- Final angle = weighted sum of bin centers using softmax probabilities
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torchvision

class Hopenet(nn.Module):
    """
    HopeNet: Head Orientation Prediction Network
    
    Architecture:
    - ResNet-50 backbone for feature extraction
    - Three separate classification heads for yaw, pitch, roll
    - Each head predicts 66-bin probability distribution (-99° to +99° in 3° increments)
    - Uses dual-loss: classification loss + regression loss for continuous angles
    """
    def __init__(self, block, layers, num_bins):
        """
        Initialize HopeNet model
        
        Args:
            block: ResNet building block (Bottleneck for ResNet-50)
            layers: List defining ResNet architecture [3,4,6,3] for ResNet-50
            num_bins: Number of angle bins (66 for -99° to +99° in 3° steps)
        """
        self.inplanes = 64
        super(Hopenet, self).__init__()
        
        # ResNet-50 backbone components
        # Initial convolution layer (7x7 conv, stride=2, reduces spatial size)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet residual blocks - progressive feature extraction
        self.layer1 = self._make_layer(block, 64, layers[0])   # 64 channels
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 128 channels
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 256 channels  
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 512 channels
        
        # Global average pooling - reduces spatial dimensions to 1x1
        self.avgpool = nn.AvgPool2d(7)
        
        # Three separate classification heads for each Euler angle
        # Each head predicts probability distribution over num_bins classes
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)    # Yaw angle head
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)  # Pitch angle head
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)   # Roll angle head

        # Vestigial layer from previous experiments (not used in current implementation)
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        # Initialize weights using He initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Build ResNet residual layer with specified number of blocks
        
        Args:
            block: ResNet building block type (Bottleneck)
            planes: Number of output channels
            blocks: Number of residual blocks in this layer
            stride: Stride for first block (for downsampling)
            
        Returns:
            Sequential layer containing residual blocks
        """
        downsample = None
        # Create downsampling layer if spatial size or channels change
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block may downsample
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # Remaining blocks maintain spatial size
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through HopeNet
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            Tuple of (yaw_logits, pitch_logits, roll_logits)
            Each logits tensor has shape [batch_size, num_bins]
            
        Forward Pass Flow:
        1. Input: 224x224x3 image
        2. Initial conv + pooling: 56x56x64
        3. ResNet layers: progressive feature extraction
        4. Global average pooling: 1x1x2048
        5. Three separate FC heads: each outputs num_bins probabilities
        """
        # Initial convolution and pooling - reduces spatial resolution
        x = self.conv1(x)      # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # [B, 64, 56, 56]

        # ResNet residual layers - hierarchical feature extraction
        x = self.layer1(x)     # [B, 256, 56, 56]
        x = self.layer2(x)     # [B, 512, 28, 28] 
        x = self.layer3(x)     # [B, 1024, 14, 14]
        x = self.layer4(x)     # [B, 2048, 7, 7]

        # Global average pooling - spatial dimensions to 1x1
        x = self.avgpool(x)    # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2048] - flatten for FC layers
        
        # Three separate classification heads
        # Each predicts probability distribution over angle bins
        pre_yaw = self.fc_yaw(x)     # [B, 66] - yaw angle bins
        pre_pitch = self.fc_pitch(x) # [B, 66] - pitch angle bins  
        pre_roll = self.fc_roll(x)   # [B, 66] - roll angle bins

        return pre_yaw, pre_pitch, pre_roll

def create_hopenet(num_bins=66):
    """
    Create HopeNet model with ResNet-50 backbone
    
    Args:
        num_bins: Number of angle bins (default 66 for -99° to +99° in 3° steps)
        
    Returns:
        HopeNet model with ResNet-50 architecture
        
    Note: Uses Bottleneck blocks with layer configuration [3,4,6,3] for ResNet-50
    """
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
    return model

if __name__ == "__main__":
    # Test the model
    model = create_hopenet(num_bins=66)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"HopeNet parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    yaw, pitch, roll = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Yaw output shape: {yaw.shape}")
    print(f"Pitch output shape: {pitch.shape}")
    print(f"Roll output shape: {roll.shape}")
    
    print("HopeNet model test completed successfully!")