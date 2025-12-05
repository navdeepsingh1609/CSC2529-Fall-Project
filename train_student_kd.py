# File: train_student_kd.py
"""
Unified Knowledge Distillation (KD) training script for UDC-SIT.

Trains the UNetStudent model using a pre-trained FrequencyAwareTeacher.
Distillation Objectives:
- Pixel Loss (Charbonnier)
- Feature Loss (Spatial + Frequency)
- Perceptual Loss (LPIPS)

Key Features:
- Supports both 'v1' and 'v2' teacher variants.
- Freezes teacher weights during training.
- Mixed-precision training (AMP).
- Google Drive integration.
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt

# Optimize for speed
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ensure MambaIR submodule is in path
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher, FrequencyAwareTeacherV2
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmplitudeLoss, FFTAmpPhaseMultiScaleLoss
from losses.pixel_loss import CharbonnierLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UNetStudent with frequency-aware KD on UDC-SIT."
    )
    # Model Configuration
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="Teacher variant: 'v1' (Amplitude) or 'v2' (Multi-Scale Phase).",
    )
    
    # Data Configuration
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing 'training' and 'validation' subfolders.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/content/dataset/UDC-SIT/training",
        help="Training data directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="/content/dataset/UDC-SIT/validation",
        help="Validation data directory.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Input patch size.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Limit training images (debug/quick run).",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Limit validation images.",
    )

    # Teacher Configuration
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default="teacher_4ch_final.pth",
        help="Path to pre-trained teacher checkpoint.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )

    # Logging & Checkpointing
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="student_kd_4ch",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/checkpoints",
        help="Drive directory for mirroring checkpoints.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Overwrite existing checkpoints and train from scratch.",
    )
    parser.add_argument(
        "--save-loss-history",
        dest="save_loss_history",
        action="store_true",
        help="Enable loss history saving (default).",
    )
    parser.add_argument(
        "--no-save-loss-history",
        dest="save_loss_history",
        action="store_false",
        help="Disable loss history saving.",
    )

    # Convenience Presets
    parser.add_argument(
        "--preset",
        type=str,
        choices=["full", "quick"],
        default=None,
        help="Configuration preset: 'quick' or 'full'.",
    )

    parser.set_defaults(save_loss_history=True, force_train=False)
    return parser.parse_args()


def maybe_apply_preset(args):
    """Applies configuration presets if explicit arguments are not provided."""
    if args.preset == "quick":
        if args.batch_size == 64:
            args.batch_size = 16
        if args.num_epochs == 20:
            args.num_epochs = 8
        if args.max_train_images is None:
            args.max_train_images = 40
        if args.max_val_images is None:
            args.max_val_images = 20
    return args


def build_dataloaders(args, device):
    """Constructs training and validation DataLoaders."""
    print("\n--- [Data] Loading Datasets ---")
    train_dataset = UDCDataset(
        args.train_dir,
        patch_size=args.patch_size,
        is_train=True
    )
    val_dataset = UDCDataset(
        args.val_dir,
        patch_size=args.patch_size,
        is_train=False
    )

    if args.max_train_images is not None:
        max_n = min(args.max_train_images, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(max_n)))
        print(f"--- [Data] Training subset: {max_n} images.")

    if args.max_val_images is not None:
        max_n = min(args.max_val_images, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(max_n)))
        print(f"--- [Data] Validation subset: {max_n} images.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"--- [Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def load_existing_checkpoint(model, candidate_paths, device, tag):
    """Attempts to load model weights from a list of candidate paths."""
    for ckpt_path in candidate_paths:
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            print(f"--- [{tag}] Resumed from checkpoint: {ckpt_path}")
            return ckpt_path
    return None


def main():
    args = parse_args()
    args = maybe_apply_preset(args)
    
    if args.data_root:
        args.train_dir = os.path.join(args.data_root, "training")
        args.val_dir = os.path.join(args.data_root, "validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.drive_checkpoint_dir, exist_ok=True)

    print("\n--- [Config] KD Training Configuration ---")
    print(f"Variant:         {args.model_variant}")
    print(f"Teacher Weights: {args.teacher_weights}")
    print(f"Data Root:       {args.data_root}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Epochs:          {args.num_epochs}")
    print(f"Device:          {device}")
    print(f"Drive Output:    {args.drive_checkpoint_dir}")
    print("------------------------------------------\n")

    if not os.path.exists(args.teacher_weights):
        raise FileNotFoundError(
            f"Teacher weights not found at {args.teacher_weights}. "
            "Please train the teacher model first."
        )

    # Define Checkpoint Paths
    prefix = args.checkpoint_prefix
    ckpt_latest_local = f"{prefix}_latest.pth"
    ckpt_final_local  = f"{prefix}_final.pth"
    ckpt_latest_drive = os.path.join(args.drive_checkpoint_dir, f"{prefix}_latest.pth")
    ckpt_final_drive  = os.path.join(args.drive_checkpoint_dir, f"{prefix}_final.pth")
    ckpt_best_drive   = os.path.join(args.drive_checkpoint_dir, f"{prefix}_best.pth")

    loss_hist_local = f"{prefix}_loss_history.npz"
    loss_hist_drive = os.path.join(args.drive_checkpoint_dir, f"{prefix}_loss_history.npz")
    loss_plot_local = f"{prefix}_loss_curves.png"
    loss_plot_drive = os.path.join(args.drive_checkpoint_dir, f"{prefix}_loss_curves.png")

    # Initialize Data
    train_loader, val_loader = build_dataloaders(args, device)

    # Initialize Teacher (Frozen)
    print(f"--- [Model] Loading Teacher ({args.model_variant})...")
    if args.model_variant == "v2":
        teacher = FrequencyAwareTeacherV2(in_channels=4, out_channels=4).to(device)
    else:
        teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(device)
    
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("--- [Model] Teacher loaded and frozen.")

    # Initialize Student
    print("--- [Model] Initializing Student...")
    student = UNetStudent(in_channels=4, out_channels=4).to(device)
    
    if not args.force_train:
        candidate_ckpts = [
            ckpt_final_drive, ckpt_final_local,
            ckpt_best_drive,
            ckpt_latest_drive, ckpt_latest_local,
        ]
        load_existing_checkpoint(student, candidate_ckpts, device, "Student")

    # Initialize Losses
    print(f"--- [Loss] Initializing losses for {args.model_variant}...")
    pixel_loss_fn     = CharbonnierLoss().to(device)
    feature_loss_fn   = nn.L1Loss().to(device)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    if args.model_variant == "v2":
        frequency_loss_fn = FFTAmpPhaseMultiScaleLoss(
            loss_weight=1.0, focus_low_freq=True, cutoff=0.25,
            lambda_amp=1.0, lambda_phase=0.5, scales=(1.0, 0.5)
        ).to(device)
    else:
        frequency_loss_fn = FFTAmplitudeLoss(
            loss_weight=1.0, focus_low_freq=True, cutoff=0.25
        ).to(device)

    # Loss Weights
    W_PIXEL   = 1.0
    W_FEATURE = 0.5
    W_FREQ    = 0.2
    W_LPIPS   = 0.1

    # Optimization
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler()

    # Training Loop
    train_loss_history = []
    val_loss_history   = []
    best_val_loss      = float("inf")

    print("--- [Train] Starting KD training loop...")

    for epoch in range(args.num_epochs):
        student.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for udc_batch, gt_batch in pbar:
            udc_batch = udc_batch.to(device, non_blocking=True)
            gt_batch  = gt_batch.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Teacher Forward (No Grad)
            with torch.no_grad():
                with autocast():
                    teacher_out_4ch, teacher_feat_spatial, _ = teacher(udc_batch)

            # Student Forward
            with autocast():
                student_out_4ch, student_features_raw = student(udc_batch)

                # 1. Pixel Loss
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)

                # 2. Spatial Feature Loss (Resize student features to match teacher)
                student_features = F.interpolate(
                    student_features_raw,
                    size=teacher_feat_spatial.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                loss_feature_spatial = feature_loss_fn(student_features, teacher_feat_spatial)

                # 3. Perceptual Loss (LPIPS on pseudo-RGB)
                pred_rgb_slice = student_out_4ch[:, :3, :, :]
                gt_rgb_slice   = gt_batch[:, :3, :, :]
                loss_lpips = perceptual_loss_fn(
                    pred_rgb_slice * 2.0 - 1.0,
                    gt_rgb_slice   * 2.0 - 1.0
                ).mean()

            # 4. Frequency Loss (Float32)
            loss_freq_out = frequency_loss_fn(
                student_out_4ch.float(),
                teacher_out_4ch.float()
            )
            loss_feature_freq = frequency_loss_fn(
                student_features.float(),
                teacher_feat_spatial.float()
            )

            # Combined Feature Loss
            loss_feature = 0.5 * loss_feature_spatial + 0.5 * loss_feature_freq
            loss_freq    = loss_freq_out

            total_loss = (W_PIXEL   * loss_pixel)   + \
                         (W_FEATURE * loss_feature) + \
                         (W_FREQ    * loss_freq)    + \
                         (W_LPIPS   * loss_lpips)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += total_loss.item()
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validation
        student.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            for udc_batch, gt_batch in pbar_val:
                udc_batch = udc_batch.to(device, non_blocking=True)
                gt_batch  = gt_batch.to(device,  non_blocking=True)

                with autocast():
                    student_out_4ch, _ = student(udc_batch)
                    loss_val = pixel_loss_fn(student_out_4ch, gt_batch)

                running_val_loss += loss_val.item()
                pbar_val.set_postfix({"val_loss": f"{loss_val.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        # Update History
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Checkpointing
        torch.save(student.state_dict(), ckpt_latest_local)
        torch.save(student.state_dict(), ckpt_latest_drive)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), ckpt_best_drive)
            print(f"New best validation loss! Saved to {ckpt_best_drive}")

        # Save History Artifacts
        if args.save_loss_history:
            np.savez(
                loss_hist_local,
                train_loss=np.array(train_loss_history, dtype=np.float32),
                val_loss=np.array(val_loss_history,   dtype=np.float32),
            )
            np.savez(
                loss_hist_drive,
                train_loss=np.array(train_loss_history, dtype=np.float32),
                val_loss=np.array(val_loss_history,   dtype=np.float32),
            )

    # Final Save
    torch.save(student.state_dict(), ckpt_final_local)
    torch.save(student.state_dict(), ckpt_final_drive)
    print(f"--- [Train] Completed. Final student model saved to {ckpt_final_drive}")

    # Plotting
    if args.save_loss_history:
        epochs = np.arange(1, len(train_loss_history) + 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, train_loss_history, label="Train")
        ax.plot(epochs, val_loss_history,   label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Student KD Training & Validation Loss")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_plot_local, dpi=150)
        plt.close(fig)

        try:
            import shutil
            shutil.copy(loss_plot_local, loss_plot_drive)
            print(f"Saved loss plot to {loss_plot_drive}")
        except Exception as e:
            print(f"Warning: Could not copy plot to Drive: {e}")


if __name__ == "__main__":
    main()
