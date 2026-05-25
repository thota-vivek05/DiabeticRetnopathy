# Diabetic Retinopathy Classification using EfficientNet-B3 with Grad-CAM Explainability

Deep learning based Computer-Aided Diagnosis (CAD) system for automated diabetic retinopathy grading using retinal fundus images.

This project explores clinically meaningful diabetic retinopathy classification using EfficientNet-B3 with transfer learning and Grad-CAM explainability across 5-class, 4-class, and 3-class diagnostic strategies.

---

# Overview

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. Early diagnosis is critical for preventing irreversible vision loss, but manual retinal screening is time-consuming, expensive, and dependent on expert ophthalmologists.

This project presents a deep learning based retinal image classification pipeline capable of:

- Automated DR severity grading
- Multi-granularity classification comparison
- Large-scale retinal image training
- Clinically interpretable AI predictions using Grad-CAM
- Efficient transfer learning with EfficientNet-B3

The model was trained on a combined retinal dataset containing approximately **143,000 fundus images** derived from:

- EyePACS
- APTOS 2019
- Messidor

---

# Key Features

- EfficientNet-B3 transfer learning architecture
- 5-class, 4-class, and 3-class DR classification
- Grad-CAM based explainable AI visualizations
- Large-scale augmented retinal dataset
- Mixed precision GPU training
- Optimized `tf.data` input pipeline
- Clinical decision-oriented class remapping
- Comparative analysis of classification granularities

---

# Problem Statement

Traditional diabetic retinopathy grading suffers from several real-world challenges:

- Severe class imbalance
- Subtle inter-class differences
- Label ambiguity between adjacent DR stages
- Lack of explainability in black-box models

This work addresses these limitations through:
- dataset augmentation,
- class consolidation strategies,
- transfer learning,
- and Grad-CAM interpretability.

---

# Model Architecture

## Backbone Network
- EfficientNet-B3 (ImageNet pretrained)

## Classification Head
- Global Average Pooling
- Dense Layer (256 units, ReLU)
- Dropout (0.5)
- Softmax Output Layer

## Explainability
- Grad-CAM visualization on final convolutional layers

---

# Dataset

This project combines multiple retinal fundus imaging datasets:

| Dataset | Purpose |
|---|---|
| EyePACS | Large-scale DR screening |
| APTOS 2019 | Clinical DR grading |
| Messidor | Retinal pathology validation |

---

# Original DR Severity Classes

| Label | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

# Classification Strategies

## 5-Class Classification
Preserves the complete clinical grading scale.

## 4-Class Classification
Merges:
- Severe DR
- Proliferative DR

into a single Vision-Threatening DR category.

## 3-Class Classification

| Class | Meaning |
|---|---|
| 0 | No DR |
| 1 | Non-Proliferative DR |
| 2 | Vision-Threatening DR |

This strategy aligns directly with practical clinical triage workflows.

---

# Data Augmentation

To improve generalization and reduce class imbalance, the following augmentations were applied:

- Random Horizontal Flip
- Random Vertical Flip
- Rotation Augmentation
- Brightness Jitter
- Contrast Jitter
- Random Zoom

---

# Training Strategy

## Phase 1 — Classification Head Training
- EfficientNet backbone frozen
- Only classification head trained

## Phase 2 — End-to-End Fine Tuning
- Last 100 EfficientNet layers unfrozen
- Reduced learning rate fine-tuning

---

# Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Batch Size | 256 |
| Learning Rate (Phase 1) | 1e-3 |
| Learning Rate (Phase 2) | 2e-5 |
| Input Resolution | 256 × 256 |
| Precision | Mixed Float16 |
| Loss Function | Categorical Cross Entropy |

---

# Results

## Performance Comparison

| Model | Accuracy | Macro F1 |
|---|---|---|
| 5-Class | 67.72% | 0.54 |
| 4-Class | 67.03% | 0.53 |
| 3-Class | **79.14%** | **0.77** |

---

# Best Performing Configuration

- EfficientNet-B3
- 3-Class DR formulation
- Grad-CAM explainability
- Large-scale augmented dataset

The 3-class strategy achieved the best balance between:
- classification accuracy,
- minority class recall,
- and clinical usefulness.

---

# Grad-CAM Explainability

Grad-CAM was used to visualize retinal regions influencing model predictions.

The model consistently focused on clinically meaningful lesion regions such as:

- Microaneurysms
- Hemorrhages
- Hard Exudates
- Neovascularization

This improves:
- interpretability,
- clinical trust,
- and explainable AI deployment.

---

# Repository Structure

```text
DiabeticRetinopathy/
│
├── notebooks/
│   ├── 01_5class_architecture_comparison.ipynb
│   ├── 02_3class_4class_experiments.ipynb
│   └── 03_gradcam_explainability.ipynb
│
├── images/
│   ├── efficientnet_architecture.png
│   ├── pipeline.png
│   ├── training_phases.png
│   ├── mbconv_block.png
│   ├── augmentation_examples.png
│   └── class_consolidation.png
│
├── results/
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   ├── class_distribution.png
│   ├── 5class_confusion_matrix.png
│   ├── 4class_confusion_matrix.png
│   ├── 3class_confusion_matrix.png
│   └── gradcam_samples/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Tech Stack

- Python
- TensorFlow / Keras
- EfficientNet
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone https://github.com/thota-vivek05/DiabeticRetinopathy.git
cd DiabeticRetinopathy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# Future Improvements

- Vision Transformer (ViT) based DR classification
- Federated learning for privacy-preserving healthcare AI
- Real-time retinal screening deployment
- Mobile inference optimization
- Ensemble learning approaches
- Clinical web application interface

---

# Research Paper

This repository accompanies our IEEE-format research paper:

### "Diabetic Retinopathy Classification Using EfficientNet-B3 with Transfer Learning and Grad-CAM Explainability"

---

# Authors

### Thota Vivek  
Department of Computer Science and Engineering  
Indian Institute of Information Technology Sri City

### Bulla Rajesh  
Department of Computer Science and Engineering  
Indian Institute of Information Technology Sri City

---

# License

This project is intended for academic research and educational purposes.