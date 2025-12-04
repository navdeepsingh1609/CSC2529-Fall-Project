# UDC-SIT: Under-Display Camera Image Restoration

This repository contains the implementation for UDC image restoration using a Frequency-Aware Teacher-Student framework.

## Key Features
- **Unified Workflow**: Single set of scripts for both Model 1 (Amplitude Loss) and Model 2 (Amplitude + Phase Loss).
- **Frequency-Aware Teacher**: Combines MambaIR spatial backbone with a Frequency Domain Block.
- **Knowledge Distillation**: Lightweight UNet Student distilled from the Teacher.
- **Google Drive Integration**: Seamless saving of checkpoints and results to Drive.

## Repository Structure
```
.
├── datasets/           # Data loading logic (UDCDataset)
├── losses/             # Loss functions (Charbonnier, Frequency, LPIPS)
├── models/             # Model definitions (Teacher, Student, MambaIR)
├── scripts/            # Utility scripts (e.g., subset creation)
├── train_teacher.py    # Main teacher training script
├── train_student_kd.py # Main student KD training script
├── testing_udc.py      # Unified evaluation script
└── Unified_Training_Testing.ipynb # Colab notebook for end-to-end runs
```

## Usage

### 1. Training the Teacher
Use `train_teacher.py` with `--model-variant` to select the configuration.

**Model 1 (Amplitude Loss):**
```bash
python train_teacher.py --model-variant v1 --preset full
```

**Model 2 (Multi-Scale Phase Loss):**
```bash
python train_teacher.py --model-variant v2 --preset full
```

### 2. Training the Student (Knowledge Distillation)
First, ensure you have a trained teacher checkpoint.

**Distill from Model 1 Teacher:**
```bash
python train_student_kd.py --model-variant v1 --teacher-weights path/to/teacher_v1.pth --preset full
```

**Distill from Model 2 Teacher:**
```bash
python train_student_kd.py --model-variant v2 --teacher-weights path/to/teacher_v2.pth --preset full
```

### 3. Evaluation
Use `testing_udc.py` to evaluate either model.

```bash
python testing_udc.py \
    --model-type student \
    --checkpoint-path path/to/student.pth \
    --test-dir /path/to/test/data
```

## Google Colab
The `Unified_Training_Testing.ipynb` notebook provides a complete environment setup and execution flow for Google Colab, including:
- Dataset extraction from Drive.
- Dependency installation.
- Training and Evaluation commands.
