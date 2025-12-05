# Frequency-Aware Image Restoration for Under-Display Cameras

A comprehensive PyTorch framework for restoring images captured by Under-Display Cameras (UDC), leveraging **Frequency-Aware Knowledge Distillation** and **MambaIR v2**.

## Overview

Under-Display Camera (UDC) technology enables full-screen displays but introduces severe image degradation due to diffraction and low light transmission through the organic LED panel. This project addresses these challenges with a progressive two-stage distillation framework:

1.  **Frequency-Aware Teacher**: A high-capacity model (MambaIR v2 + Frequency Branch) trained on full-resolution images to learn global degradation patterns.
2.  **Lightweight Student**: A U-Net student distilled from the teacher using multi-domain losses (Spatial, Frequency, and Perceptual).

## Key Features

*   **Dual-Domain Processing**: Explicitly handles diffraction artifacts in both spatial and frequency domains.
*   **MambaIR v2 Integration**: Utilizes State-Space Models (SSM) with Attentive State-Space Modules (ASSM) for efficient long-range dependency modeling.
*   **Knowledge Distillation**: Compresses the heavy teacher model into a lightweight student suitable for mobile deployment.
*   **Progressive Variants**:
    *   **Method 1 (Baseline)**: Spectral Amplitude Distillation.
    *   **Method 2 (Proposed)**: Multi-Scale Phase-Aware Distillation with Gated Fusion.

## Repository Structure

```
├── models/
│   ├── basic_block.py       # Basic building blocks (Conv, ResBlock)
│   ├── frequency_block.py   # Frequency Domain Processing Blocks
│   ├── mambair_teacher.py   # Teacher: MambaIR v2 + Frequency Branch
│   └── unet_student.py      # Student: U-Net + KD Heads
├── losses/
│   ├── frequency_loss.py    # Amplitude and Phase-Aware FFT Losses
│   └── pixel_loss.py        # Charbonnier Loss
├── datasets/
│   └── udc_dataset.py       # UDC-SIT Dataset Loader (Numpy format)
├── train_teacher.py         # Teacher training script
├── train_student_kd.py      # Student distillation script
├── testing_udc.py           # Evaluation script (Full-Res & Tiled)
└── Unified_Training_Testing.ipynb # Complete Colab Workflow
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
pip install timm mamba-ssm causal-conv1d
```

### 2. Training

**Train the Teacher (Stage 1):**
```bash
python train_teacher.py \
    --model-variant v2 \
    --data-root ./dataset/UDC-SIT \
    --batch-size 8 \
    --num-epochs 22
```

**Train the Student (Stage 2 - Distillation):**
```bash
python train_student_kd.py \
    --model-variant v2 \
    --teacher-weights checkpoints/teacher_v2_final.pth \
    --data-root ./dataset/UDC-SIT \
    --batch-size 64
```

### 3. Evaluation

Evaluate the trained student model using either full-resolution inference or patch-based tiling:

```bash
python testing_udc.py \
    --model-type student \
    --checkpoint-path checkpoints/student_kd_final.pth \
    --data-root ./dataset/UDC-SIT/testing \
    --eval-mode full  # or 'tiled' for memory efficiency
```

## Method Comparison

| Feature | Method 1 (Baseline) | Method 2 (Proposed) |
| :--- | :--- | :--- |
| **Fusion Mechanism** | Direct Concatenation | Gated Fusion (Spatial-guided) |
| **Frequency Loss** | Amplitude Only (L1) | Multi-Scale Amplitude + Phase |
| **Student Architecture** | Standard U-Net | U-Net + High-Res Skip Frequency Block |

## Methodology

For a detailed mathematical formulation of the loss functions and architectural decisions, please refer to `Methodology.docx` or `methodology.tex`.

## Acknowledgements

This project builds upon:
*   **MambaIR**: "A Simple Baseline for Image Restoration with State-Space Model"
*   **LPIPS**: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
