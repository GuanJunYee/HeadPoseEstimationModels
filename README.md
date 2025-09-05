# Head Pose Deep Learning

A comprehensive implementation and comparison of deep learning models for 3D head pose estimation using the AFLW2000 dataset. This project implements and compares three different approaches: HopeNet from scratch, fine-tuned HopeNet, and a custom CNN architecture.

## Project Overview

Head pose estimation is a crucial computer vision task with applications in human-computer interaction, driver monitoring, and augmented reality. This project implements state-of-the-art head pose estimation techniques using PyTorch, focusing on the dual-loss methodology introduced in HopeNet.

### Key Features
- **Multiple Model Architectures**: HopeNet, Custom CNN, and MobileNetV2 implementations
- **Dual-Loss Methodology**: Combines classification and regression losses for improved accuracy
- **Comprehensive Evaluation**: Detailed comparison across different model architectures
- **Production-Ready Code**: Well-documented, modular codebase with proper error handling

## Results Summary

| Model | Method | Validation MAE | Test MAE | Parameters |
|-------|--------|----------------|----------|------------|
| HopeNet (Fine-tuned) | Transfer Learning | 4.20° | 5.12° | ~23M |
| HopeNet (From Scratch) | End-to-end Training | TBD | TBD | ~23M |
| Custom CNN | Lightweight Architecture | TBD | TBD | ~12M |

*MAE = Mean Absolute Error across yaw, pitch, and roll angles*

## Architecture

### HopeNet (ResNet-50 Backbone)
- **Backbone**: ResNet-50 pre-trained on ImageNet
- **Heads**: Three separate classification heads for yaw, pitch, roll
- **Innovation**: Dual-loss function combining classification and regression
- **Angle Binning**: 66 bins from -99° to +99° in 3° increments

### Custom CNN
- **Design**: Lightweight 4-layer CNN with progressive channel expansion
- **Architecture**: 32→64→128→256 channels with batch normalization
- **Methodology**: Adopts HopeNet's dual-loss approach
- **Advantage**: Fewer parameters for faster training and inference

### MobileNetV2
- **Backbone**: MobileNetV2 for mobile deployment
- **Optimization**: Designed for resource-constrained environments
- **Trade-off**: Balanced accuracy vs computational efficiency

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Head-Pose-Deep-Learning.git
cd Head-Pose-Deep-Learning

# Create virtual environment
python -m venv head_pose_env
source head_pose_env/bin/activate  # On Windows: head_pose_env\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install opencv-python pillow numpy scipy
pip install scikit-learn matplotlib seaborn
pip install tqdm
```

## Dataset Setup

### AFLW2000 Dataset
1. Download the AFLW2000 dataset from the official source
2. Extract to `data/AFLW2000/` directory
3. Ensure the structure follows:
```
data/
├── AFLW2000/
│   ├── image00002.jpg
│   ├── image00002.mat
│   ├── image00004.jpg
│   ├── image00004.mat
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Data Splits Creation
```bash
python create_data_splits.py
```

## Usage

### Training Models

#### 1. Train HopeNet from Scratch
```bash
python training/train_hopenet.py
```

#### 2. Fine-tune Pre-trained HopeNet
```bash
python training/finetune_hopenet.py
```

#### 3. Train Custom CNN
```bash
python training/train_custom_cnn.py
```

#### 4. Train MobileNetV2
```bash
python training/train_mobilenetv2.py
```

### Evaluation

#### Test Individual Models
```bash
# Test fine-tuned HopeNet
python evaluation/test_finetuned_hopenet.py

# Test all models comparatively
python evaluation/test_all_models.py
```

#### Debug and Analysis
```bash
# Debug model checkpoints
python debug_custom_cnn_checkpoint.py

# Analyze data format
python test_data_format.py

# Count dataset samples
python count_samples.py
```

## Methodology

### Dual-Loss Function
The project implements HopeNet's innovative dual-loss approach:

```python
# Classification Loss (discrete bins)
classification_loss = CrossEntropyLoss(logits, binned_labels)

# Regression Loss (continuous angles)
predicted_angles = softmax_expected_value(logits)
regression_loss = MSELoss(predicted_angles, continuous_labels)

# Combined Loss
total_loss = classification_loss + α * regression_loss
```

### Angle Conversion
Continuous angles are converted to discrete bins and back:
- **Binning**: 66 bins from -99° to +99° (3° increments)
- **Expected Value**: Weighted sum using softmax probabilities
- **Formula**: `angle = Σ(P(bin_i) × bin_center_i)`

### Data Augmentation
- **Horizontal Flipping**: 50% probability with angle correction
- **Blur Augmentation**: 5% probability for robustness
- **Face Cropping**: Loose cropping with 20% padding using 2D landmarks

## Technical Details

### Model Architectures

#### HopeNet Forward Pass
```python
# Feature extraction
features = resnet50_backbone(image)  # [B, 2048, 7, 7]
features = global_avg_pool(features)  # [B, 2048]

# Three classification heads
yaw_logits = fc_yaw(features)      # [B, 66]
pitch_logits = fc_pitch(features)  # [B, 66]
roll_logits = fc_roll(features)    # [B, 66]

return yaw_logits, pitch_logits, roll_logits
```

#### Custom CNN Architecture
```python
# Progressive feature extraction
x = conv_block_1(x)  # 224×224×3 → 112×112×32
x = conv_block_2(x)  # 112×112×32 → 56×56×64
x = conv_block_3(x)  # 56×56×64 → 28×28×128
x = conv_block_4(x)  # 28×28×128 → 14×14×256

# Classification heads
features = fc_layers(flatten(x))  # [B, 256]
yaw_logits = fc_yaw(features)     # [B, 66]
# ... similar for pitch and roll
```

### Training Configuration
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Batch Size**: 16
- **Optimizer**: Adam with weight decay (1e-4)
- **Early Stopping**: 20 epochs patience
- **Gradient Clipping**: Max norm = 1.0

## Evaluation Metrics

### Primary Metrics
- **Overall MAE**: Mean Absolute Error across all three angles
- **Per-Angle MAE**: Individual errors for yaw, pitch, roll
- **Validation vs Test**: Generalization assessment

### Visualization
The project generates comprehensive training curves and analysis:
- Loss progression (training vs validation)
- MAE progression by angle
- Learning rate scheduling
- Best model performance markers

## File Structure

```
Head-Pose-Deep-Learning/
├── models/
│   ├── hopenet.py          # HopeNet architecture
│   ├── custom_cnn.py       # Custom CNN implementation
│   └── mobilenetv2.py      # MobileNetV2 adaptation
├── training/
│   ├── train_hopenet.py    # HopeNet from scratch
│   ├── finetune_hopenet.py # HopeNet fine-tuning
│   ├── train_custom_cnn.py # Custom CNN training
│   └── train_mobilenetv2.py
├── evaluation/
│   ├── test_all_models.py  # Comprehensive evaluation
│   └── test_finetuned_hopenet.py
├── utils/
│   ├── datasets.py         # AFLW2000 data loader
│   └── utils.py           # Utility functions
├── data/
│   ├── AFLW2000/          # Dataset images and annotations
│   └── splits/            # Train/val/test splits
└── results/               # Model checkpoints and results
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

1. **HopeNet Paper**: "Fine-Grained Head Pose Estimation Without Keypoints" by Ruiz et al. (2018).
2. **AFLW2000 Dataset**: by Adly (2024).

---

**If you find this project helpful, please consider giving it a star!**
