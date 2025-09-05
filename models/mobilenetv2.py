# models/mobilenetv2.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MobileNetV2HeadPose(nn.Module):
    """MobileNetV2 with regularization for head pose estimation"""
    
    def __init__(self, num_bins=66, pretrained=True, dropout_rate=0.5):
        super(MobileNetV2HeadPose, self).__init__()
        
        # Load pre-trained MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Remove the final classification layer
        num_features = self.backbone.classifier[1].in_features  # 1280 features
        self.backbone.classifier = nn.Identity()
        
        self.num_bins = num_bins
        
        # Add regularized classification heads with dropout
        self.fc_yaw = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_bins)
        )
        self.fc_pitch = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_bins)
        )
        self.fc_roll = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_bins)
        )
        
        # Initialize classification heads
        self._initialize_classification_heads()
    
    def _initialize_classification_heads(self):
        """Initialize classification head weights"""
        for module in [self.fc_yaw, self.fc_pitch, self.fc_roll]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using MobileNetV2
        features = self.backbone(x)  # Shape: [batch, 1280]
        
        # Classification outputs with dropout regularization
        yaw_logits = self.fc_yaw(features)      # Shape: [batch, 66]
        pitch_logits = self.fc_pitch(features)  # Shape: [batch, 66]
        roll_logits = self.fc_roll(features)    # Shape: [batch, 66]
        
        return yaw_logits, pitch_logits, roll_logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

def create_mobilenetv2_headpose(num_bins=66, pretrained=True, dropout_rate=0.5):
    """Create regularized MobileNetV2 head pose estimation model"""
    model = MobileNetV2HeadPose(num_bins=num_bins, pretrained=pretrained, dropout_rate=dropout_rate)
    return model

if __name__ == "__main__":
    # Test the model
    model = create_mobilenetv2_headpose(num_bins=66, pretrained=True, dropout_rate=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_parameters()
    model_size_mb = model.get_model_size_mb()
    
    print(f"MobileNetV2 Head Pose Model (Regularized):")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print(f"  Dropout rate: 0.5")
    
    # Test forward pass
    model.eval()  # Test in eval mode to see dropout effect
    dummy_input = torch.randn(2, 3, 224, 224)
    yaw_logits, pitch_logits, roll_logits = model(dummy_input)
    
    print(f"\nModel test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Yaw output shape: {yaw_logits.shape}")
    print(f"  Pitch output shape: {pitch_logits.shape}")
    print(f"  Roll output shape: {roll_logits.shape}")
    
    # Compare with other models
    print(f"\nModel Size Comparison:")
    print(f"  Custom CNN: ~1M parameters (~4 MB)")
    print(f"  MobileNetV2: {total_params/1e6:.1f}M parameters (~{model_size_mb:.1f} MB)")
    print(f"  ResNet50/HopeNet: ~25M parameters (~100 MB)")
    print(f"  Efficiency: {25/total_params*1e6:.1f}x smaller than ResNet50!")
    
    print("\nRegularized MobileNetV2 model test completed successfully!")