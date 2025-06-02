# Age-Variant Face Matching System

A comprehensive deep learning system that identifies the same individual across different ages by combining age prediction with face matching capabilities.

## 🎯 Project Overview

This system addresses the challenging task of recognizing the same person at different ages by:
- **Age Prediction**: Estimating the age of individuals in facial images
- **Face Matching**: Determining if two faces belong to the same person
- **Age-Variant Analysis**: Combining both models to handle facial changes over time

## 🏗️ System Architecture

```
├── age/                    # Age estimation module
│   ├── config/            # Configuration management
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Age prediction model architectures
│   ├── training/          # Training utilities and loss functions
│   ├── inference/         # Age prediction inference
│   └── utils/             # Utility functions
├── face/                  # Face matching module
│   ├── config/            # Configuration management
│   ├── data/              # Face pair dataset creation
│   ├── models/            # Siamese network for face matching
│   ├── training/          # Training pipeline
│   ├── inference/         # Face matching inference
│   └── utils/             # Visualization and utilities
├── main.py               # Integrated pipeline
└── requirements.txt      # Dependencies
```
Notice "Checkpoints here https://drive.google.com/drive/folders/1kdN1REJbHtup5S0G1si6Msk5t6wefRh0?usp=drive_link"
## 🚀 Quick Start 

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd age-variant-face-matching

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Age A: UTKFace Dataset (Recommended)
```bash
# Download UTKFace dataset
# Place extracted images in: age/Datasets/utkcropped/
```

#### Face B: Use Built-in Datasets
```bash
# The system will automatically download LFW or Olivetti faces
python face/main.py 
```

### 3. Training Models

#### Age Estimation Model
```bash
python age/main.py --epochs 50 --batch_size 128
```

#### Face Matching Model
```bash
python face/main.py --mode full
```

### 4. Running the Integrated System

```bash
python main.py --image1 img1.jpg --image2 img2.jpg \
               --age_model age/checkpoints/best_model.pt \
               --face_model face/checkpoints/best_model.pth \
               --save_viz result.jpg

```

## 📊 Dataset Information

### Primary Dataset: UTKFace
- **Size**: 20,000+ facial images
- **Age Range**: 0-116 years
- **Demographics**: Multiple ethnicities and genders
- **Format**: [age]_[gender]_[race]_[date].jpg

### Datasets
- **LFW (Labeled Faces in the Wild)**: Real-world face images

## 🧠 Model Architectures

### Age Estimation Model
- **Backbone**: ResNet50
- **Input**: 224×224 RGB images
- **Output**: Single age value (regression)
- **Loss Function**: L1 Loss (MAE) with optional variants

### Face Matching Model
- **Architecture**: Siamese Network with ResNet50 backbone
- **Embedding Dimension**: 512
- **Loss Function**: Contrastive Loss
- **Distance Metric**: Euclidean distance and cosine similarity

## 📈 Performance Metrics

### Age Estimation
- **MAE (Mean Absolute Error)**: Primary metric
- **Accuracy@5**: Percentage within 5 years
- **R²**: Coefficient of determination

### Face Matching
- **AUC**: Area Under Curve
- **Accuracy**: Classification accuracy
- **Distance/Similarity Thresholds**: Optimized thresholds

### Integrated System
- **Age-Aware Accuracy**: Accuracy considering age differences
- **Confidence Scoring**: Multi-factor confidence estimation

## 🛠️ Configuration

### Age Model Configuration
```python
# age/config/config.py
config = {
    'img_size': 224,
    'batch_size': 128,
    'lr': 0.0001,
    'epochs': 50,
    'model_name': 'resnet',
    'loss_function': 'l1'
}
```

### Face Model Configuration
```python
# face/config/config.json
{
    "embedding_dim": 512,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_epochs": 30,
    "distance_threshold": 0.5
}
```

## 📝 System Capabilities

### ✅ Strengths
1. **Robust Age Prediction**: Handles diverse age ranges (0-85+ years)
2. **Multi-Demographic Support**: Works across ethnicities and genders
3. **Age-Aware Matching**: Adjusts thresholds based on predicted age differences
4. **Flexible Architecture**: Supports multiple model backbones
5. **Comprehensive Evaluation**: Multiple metrics and confidence scoring
6. **Real-time Inference**: Optimized for practical deployment

### ⚠️ Limitations
1. **Dataset Dependency**: Performance varies with training data quality
2. **Extreme Age Gaps**: Challenging for differences >40 years
3. **Image Quality**: Sensitive to blur, lighting, and resolution
4. **Pose Variations**: Performance degrades with extreme poses
5. **Occlusions**: Struggles with partially covered faces
6. **Cross-ethnic Performance**: May have bias depending on training data

## 🔬 Evaluation

### Running Evaluation
```bash
# Age model evaluation
cd age
python main.py --eval_only --resume checkpoints/best_model.pt

# Face model evaluation
cd face
python main.py --mode evaluate --model_path checkpoints/best_model.pth

# Integrated system evaluation
python main.py --image1 test1.jpg --image2 test2.jpg --save_viz results.png
```


## 📚 Dependencies

- Python 3.8+
- PyTorch 1.10+
- torchvision
- OpenCV
- PIL/Pillow
- matplotlib
- seaborn
- scikit-learn
- pandas
- numpy
- tqdm

