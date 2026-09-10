# 🧠 NeuroSense

**Brain Tumor MRI Segmentation using Deep Learning**

NeuroSense is a Medical AI project focused on automatic brain tumor segmentation from MRI scans using deep learning and computer vision techniques.

The goal of this project is to develop an end-to-end segmentation pipeline that can identify tumor regions at the pixel level from brain MRI images.

The project is built with PyTorch and follows a complete deep learning workflow:

- Data loading
- Medical image preprocessing
- Dataset pipeline
- U-Net architecture implementation
- Training pipeline
- Validation and evaluation
- Model checkpointing
- Inference
- Visualization and analysis

> ⚠️ This project is for research and educational purposes only and is not intended for clinical diagnosis.

---

## 📌 Problem Overview

Brain tumor segmentation is an important task in medical image analysis.

Manual tumor annotation requires expert radiologists and is time-consuming.

Deep learning-based segmentation models can assist researchers and clinicians by automatically identifying tumor regions in MRI scans.

This project focuses on binary segmentation:

| Label | Meaning |
|---|---|
| 0 | Background |
| 1 | Tumor region |

The model receives an MRI slice and predicts a binary tumor mask.

---

## 🚀 Current Pipeline

```text
MRI Dataset (.mat)
        ↓
Dataset Loader
        ↓
Medical Image Preprocessing
        ↓
PyTorch DataLoader
        ↓
U-Net Segmentation Model
        ↓
Training Pipeline
        ↓
Checkpoint Saving
        ↓
Inference Predictor
        ↓
Visualization
```

---

## 📂 Dataset

The dataset consists of MATLAB `.mat` MRI files.

Each file contains:

```text
cjdata/
├── image
└── tumorMask
```

### Input — MRI Image

| Property | Value |
|---|---|
| Shape | 512 × 512 |
| dtype | int16 |

### Target — Tumor Mask

| Property | Value |
|---|---|
| Shape | 512 × 512 |
| dtype | uint8 |
| Values | 0 → Background, 1 → Tumor |

---

## 🔄 Dataset Pipeline

Implemented a custom PyTorch Dataset: `BrainTumorDataset`

**Responsibilities:**

- Reading `.mat` files using `h5py`
- Extracting MRI images
- Extracting tumor masks
- Applying preprocessing
- Returning tensors ready for training

**Output — single sample:**

```text
Image: [1, 224, 224]
Mask:  [1, 224, 224]
```

**Output — batch:**

```text
Images: [B, 1, 224, 224]
Masks:  [B, 1, 224, 224]
```

---

## 🧹 Preprocessing

A custom preprocessing module was implemented: `BrainMRIProcessor`

### Image Normalization

Only brain pixels are considered:

```python
brain_mask = image != 0
```

Mean and standard deviation are calculated from brain pixels. Background remains unchanged.

Normalization:

```text
normalized = (image - mean) / std
```

### Tensor Conversion

Pipeline:

```text
NumPy
  ↓
Torch Tensor
  ↓
Add Channel Dimension
  ↓
Resize
```

### Image Resizing

```text
MRI: 512×512 → 224×224
Interpolation: Bilinear
```

### Mask Resizing

```text
Mask: 512×512 → 224×224
Interpolation: Nearest Neighbor
```

Nearest interpolation is used because masks contain discrete labels.

---

## 🧠 Model Architecture

### U-Net

The segmentation model is a custom implementation of U-Net.

U-Net is widely used for medical image segmentation because of its encoder–decoder structure and skip connections.

**Architecture:**

```text
Input (1 channel)
   ↓
Encoder:    64 → 128 → 256 → 512
   ↓
Bottleneck: 1024
   ↓
Decoder:    512 → 256 → 128 → 64
   ↓
Output
```

### Double Convolution Block

Each block contains:

```text
Conv2d → BatchNorm → ReLU
Conv2d → BatchNorm → ReLU
```

### Skip Connections

Feature maps from the encoder are concatenated with decoder features:

```text
Encoder features + Decoder features
              ↓
     Better spatial information
```

---

## 🎯 Loss Function

Implemented: **Dice Loss**

Dice Loss is suitable for medical segmentation because tumor regions usually occupy a small portion of the image.

**Formula:**

```text
Dice = 2 * Intersection / (Prediction + Target)
Loss = 1 - Dice
```

---

## 📊 Evaluation Metrics

Implemented metrics:

- **Dice Score** — Measures overlap between prediction and ground truth. Higher is better.
- **IoU (Intersection over Union)** — Measures segmentation overlap. Higher is better.

---

## 🏋️ Training Pipeline

Implemented custom trainer: `Trainer`

**Features:**

- Training loop
- Validation loop
- Metric calculation
- Device management
- Checkpoint saving

**Configuration:**

| Item | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Device | CUDA if available |

---

## 💾 Checkpoint System

Best model is saved based on validation Dice score.

**Saved information:**

```python
{
    epoch,
    model_state_dict,
    optimizer_state_dict,
    validation_dice
}
```

**Current checkpoint:** `models/best_model.pth`

---

## 📈 Current Results

Current baseline model:

| Item | Value |
|---|---|
| Epochs | 10 |
| Train | 2144 samples |
| Validation | 1532 samples |
| Test | 1532 samples |

### Validation Performance

Current best validation results:

| Metric | Value |
|---|---|
| Validation Dice | ≈ 0.766 |
| Validation IoU | ≈ 0.628 |

The model successfully learned to localize tumor regions and produce binary segmentation masks.

---

## 🔍 Inference Pipeline

Implemented: `Predictor`

**Responsibilities:**

- Loading trained model
- Switching model to evaluation mode
- Running inference
- Applying sigmoid
- Applying threshold
- Returning binary segmentation mask

**Example output:**

```text
Prediction shape: [1, 1, 224, 224]
Unique values:    tensor([0., 1.])
```

---

## 🖼️ Visualization

Implemented visualization module:

```text
visualization/
└── visualize.py
```

**Current visualization:**

```text
MRI | Ground Truth Mask | Prediction Mask
```

**Functions:**

- `tensor_to_numpy()`
- `create_segmentation_figure()`
- `show_segmentation()`
- `save_segmentation()`

---

## 🔄 Current Development Stage

Currently working on: **Full Dataset Visualization**

**Goal:** Generate segmentation visualizations for:

- Validation dataset
- Test dataset

**Output:**

```text
outputs/
├── validation/
│   ├── sample_0000.png
│   ├── sample_0001.png
│   └── ...
└── test/
    ├── sample_0000.png
    ├── sample_0001.png
    └── ...
```

---

## 🛣️ Roadmap

### Phase 1 — Visualization and Analysis

- [ ] Complete dataset visualization
- [ ] Add MRI + Prediction overlay
- [ ] Add MRI + Ground Truth overlay
- [ ] Add per-sample Dice score
- [ ] Add per-sample IoU score

### Phase 2 — Model Evaluation

- [ ] Full validation evaluation
- [ ] Final test evaluation
- [ ] Generate evaluation report
- [ ] Analyze failure cases

### Phase 3 — Error Analysis

Analyze:

- False positives
- False negatives
- Under segmentation
- Over segmentation
- Small tumor failures
- Boundary errors

### Phase 4 — Training Improvements

Experiment with:

- Dice + BCE Loss
- Data augmentation
- Learning rate scheduler
- Early stopping
- Better preprocessing

### Phase 5 — Advanced Models

Possible future experiments:

- Attention U-Net
- U-Net with pretrained encoder
- ResUNet
- Vision Transformer based segmentation models

---

## 🧪 Future Evaluation Improvements

Additional medical segmentation metrics:

- Precision
- Recall / Sensitivity
- Specificity
- Hausdorff Distance (HD95)

---

## 🏗️ Project Structure

```text
NeuroSense/
└── src/
    └── current/
        ├── datasets/
        │   └── brain_tumor.py
        ├── utils/
        │   └── preprocessing.py
        ├── models/
        │   └── unet.py
        ├── losses/
        │   └── dice_loss.py
        ├── training/
        │   ├── trainer.py
        │   └── train.py
        ├── inference/
        │   └── predictor.py
        └── visualization/
            └── visualize.py
```

---

## 🛠️ Technologies

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch, TorchVision, NumPy |
| Medical Imaging | h5py, MATLAB `.mat` file processing |
| Visualization | Matplotlib |
| Development | Python, CUDA, Git |

---

## 👨‍💻 Author

**Mani Soltankhah**
Computer Engineering Student

Interested in:

- Artificial Intelligence
- Medical AI
- Computer Vision
- Deep Learning

---

## ⭐ Project Status

Current status:

- 🟢 Baseline U-Net trained successfully
- 🟢 Validation pipeline completed
- 🟢 Checkpoint system implemented
- 🟢 Predictor implemented
- 🟡 Dataset-wide visualization in progress
- 🟡 Advanced evaluation and improvements coming next