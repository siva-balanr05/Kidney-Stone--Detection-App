# MODEL_DOCUMENTATION.md — KidneyAI

## Overview

This document describes the three machine learning models used in the KidneyAI system.
All models are trained on the CT kidney dataset (11,069 images, 3 classes: Cyst/Normal/Tumor).

**IMPORTANT:** Metrics in this document are placeholders until actual training is completed.
Run training scripts and replace these sections with real results.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | CT Kidney Dataset (Kaggle) |
| Total Images | 11,069 |
| Format | JPEG |
| Image Size | Variable (resized during preprocessing) |
| Classes | Cyst (3,709), Normal (5,077), Tumor (2,283) |
| Class Balance | Imbalanced (Normal dominant) |
| Patient IDs | Not available |
| Split Strategy | Stratified random split (image-level) |

### Dataset Split

| Set | Size | % |
|-----|------|---|
| Training | ~7,748 | 70% |
| Validation | ~1,660 | 15% |
| Test | ~1,661 | 15% |

**Limitation:** Because no patient identifiers are available, images were split randomly.
There is a theoretical risk that images from the same scan could appear in both train and test,
though the dataset appears to use distinct images per class.

---

## Model A: Custom CNN

### Architecture

```
Input (150, 150, 3)
    ↓
Conv2D(32, 3×3, relu, same padding)
    ↓
BatchNormalization
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, relu, same padding)
    ↓
BatchNormalization
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3, relu, same padding)
    ↓
BatchNormalization
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(256, 3×3, relu, same padding)
    ↓
BatchNormalization
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu)
    ↓
Dropout(0.5)
    ↓
Dense(3, softmax)
    ↓
Output: [P(Cyst), P(Normal), P(Tumor)]
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 150 × 150 × 3 |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | Up to 30 (EarlyStopping patience=5) |
| Augmentation | Rotation±15°, Width/Height shift±10%, HFlip, Zoom±10% |
| Class Weighting | Yes (inverse frequency) |

### Preprocessing (CNN)
1. Load image with Pillow
2. Resize to 150 × 150 pixels
3. Convert to RGB
4. Convert to numpy array
5. Normalize: divide by 255.0
6. Add batch dimension

### Grad-CAM Target Layer
- Last convolutional layer: `conv2d_3`
- Applies for CNN classification branch only

### Model Files
```
model_artifacts/cnn/
├── kidney_cnn.keras       # Saved model
├── cnn_history.json       # Training loss/accuracy curves
├── cnn_metrics.json       # Test set evaluation metrics
├── class_indices.json     # {Cyst:0, Normal:1, Tumor:2}
└── version.json           # Model version string
```

### Evaluation (Run training to populate)
> Train the model using: `python ml/train_cnn.py`
> Metrics will be saved to model_artifacts/cnn/cnn_metrics.json

---

## Model B: ResNet50 Transfer Learning

### Architecture

```
Input (224, 224, 3)
    ↓
ResNet50 Base (frozen, ImageNet weights)
    ↓ [2048 feature maps]
GlobalAveragePooling2D
    ↓ [2048-dim vector]
Dense(256, relu)
    ↓
Dropout(0.4)
    ↓
Dense(3, softmax)
    ↓
Output: [P(Cyst), P(Normal), P(Tumor)]
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 |
| Base Model | ResNet50 (ImageNet) |
| Phase 1 | Freeze base, train head (20 epochs) |
| Phase 2 | Unfreeze last 30 layers, fine-tune (lr=1e-5) |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |

### Preprocessing (ResNet50)
1. Load image with Pillow
2. Resize to 224 × 224 pixels
3. Convert to RGB
4. Convert to numpy array
5. Apply `tf.keras.applications.resnet50.preprocess_input`
   (subtracts ImageNet mean, no /255 normalization — this is handled internally)

**IMPORTANT:** Do NOT use /255 normalization for ResNet50. Use preprocess_input only.

### Grad-CAM Target Layer
- Layer name: `conv5_block3_out` (last ResNet50 conv block)

### Model Files
```
model_artifacts/resnet/
├── kidney_resnet.keras    # Saved model
├── resnet_history.json    # Training curves
├── resnet_metrics.json    # Test metrics
└── version.json
```

### Evaluation (Run training to populate)
> Train the model using: `python ml/train_resnet.py`

---

## Model C: ResNet50 Feature Extractor + Random Forest

### Pipeline

```
Input Image (224, 224, 3)
    ↓
ResNet50 Base (ImageNet, no top layer)
    ↓
GlobalAveragePooling2D
    ↓
Feature Vector (2048 dimensions)
    ↓
StandardScaler (fit on train only)
    ↓
RandomForestClassifier(n_estimators=200, random_state=42)
    ↓
Output: [P(Cyst), P(Normal), P(Tumor)]
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Feature Extractor | ResNet50 base (ImageNet, no top) |
| Feature Dimensions | 2048 |
| Scaler | StandardScaler |
| n_estimators | 200 |
| max_features | 'sqrt' |
| random_state | 42 |
| class_weight | 'balanced' |

### Important Notes
- Feature extractor (ResNet50 base) is frozen (ImageNet weights, not fine-tuned)
- StandardScaler is fit ONLY on training features
- RF hyperparameters tuned on validation data, not test data
- Grad-CAM is NOT applicable to Random Forest directly
  - For explainability, Grad-CAM from the CNN model is used instead
  - This is clearly labeled in the UI

### Model Files
```
model_artifacts/random_forest/
├── rf_model.joblib        # Trained RandomForest
├── scaler.joblib          # Fitted StandardScaler
├── rf_metrics.json        # Test metrics
├── train_features.npy     # Extracted train features (saved for reference)
├── val_features.npy
├── test_features.npy
└── version.json
```

### Run Training
```bash
# Step 1: Extract features
python ml/extract_resnet_features.py

# Step 2: Train RF
python ml/train_random_forest.py
```

---

## Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) is available for:
- Custom CNN (target layer: conv2d_3)
- ResNet50 classification model (target layer: conv5_block3_out)

**NOT available for Random Forest** — tree-based models do not have differentiable activations.

The Grad-CAM visualization shows which regions of the image the neural network focused on
when making its prediction. This is a model interpretation tool, not a medical annotation.

> "Model attention visualization — not a confirmed anatomical stone or lesion location."

---

## Evaluation Methodology

All three models are evaluated on the SAME held-out test set (15% = ~1,661 images).

Metrics computed:
- Accuracy
- Per-class Precision, Recall, F1-Score
- Macro-averaged metrics
- Confusion Matrix (3×3)
- ROC-AUC (one-vs-rest)

Run `python ml/evaluate_models.py` to generate comparison results in `model_artifacts/evaluation/comparison.json`.

---

## Reproducibility

To reproduce training:
1. Use Python 3.11, TensorFlow 2.15, scikit-learn 1.8
2. Set random seed: RANDOM_SEED = 42 in ml/config.py
3. Run scripts in order: train_cnn.py → train_resnet.py → extract_resnet_features.py → train_random_forest.py → evaluate_models.py

---

## Limitations

1. **No patient-level split** — images split randomly; same-scan images may appear across splits
2. **Dataset scope** — trained on CT kidney images only; performance on ultrasound/MRI is unknown
3. **Uncalibrated probabilities** — softmax outputs are not calibrated; do not interpret as clinical confidence
4. **Class imbalance** — Normal class has 2.2× more images than Tumor; addressed by class weighting
5. **Grad-CAM limitations** — heatmaps show model attention, not ground-truth anatomy
6. **Not clinically validated** — no regulatory evaluation; for research and education only
7. **Domain shift** — ImageNet pretraining may not optimally transfer to medical imaging

---

## Disclaimer

This system is intended for research and educational purposes only.
It is NOT a clinically validated diagnostic tool.
It is NOT a substitute for professional medical diagnosis or clinical decision-making.
All predictions should be interpreted by qualified medical professionals.
