# File: train_student_kd.py
"""
Unified student KD training script for UDC-SIT.

- Trains UNetStudent with:
    * Pixel loss (Charbonnier)
    * Feature KD (spatial + freq via FFTAmpPhaseMultiScaleLoss: amp + phase)
    * LPIPS on pseudo-RGB
- Teacher is frozen FrequencyAwareTeacher, loaded from --teacher-weights.

You can switch between quick and full KD via:
    * --train-dir / --val-dir
    * --max-train-images / --max-val-images
    * --batch-size, --num-epochs, --preset quick/full

Outputs:
    * <checkpoint_prefix>_latest.pth
    * <checkpoint_prefix>_final.pth
    * <checkpoint_prefix>_loss_history.npz
    * <checkpoint_prefix>_loss_curves.png
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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Make sure external MambaIR is visible
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent
from losses.frequency_loss import FFTAmpPhaseMultiScaleLoss
from losses.pixel_loss import CharbonnierLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UNetStudent with frequency-aware KD on UDC-SIT."
    )

    # Data
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Base dataset directory containing 'training' and 'validation' subfolders. If set, overrides --train-dir/--val-dir.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/content/dataset/UDC-SIT/training",
        help="Training data root containing 'input' and 'GT' subfolders.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="/content/dataset/UDC-SIT/validation",
        help="Validation data root containing 'input' and 'GT' subfolders.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size used by UDCDataset.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="If set, limit number of training images (quick KD).",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="If set, limit number of validation images.",
    )

    # Teacher / Student
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default="teacher_4ch_final.pth",
        help="Path to trained teacher checkpoint (.pth).",
    )

    # Training hyperparams
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for student training.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of KD epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    # Checkpointing / logging
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="student_kd_4ch",
        help="Prefix for saved checkpoints and logs.",
    )
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/checkpoints",
        help="Google Drive directory where checkpoints & logs are mirrored.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Ignore existing student checkpoints and train from scratch/full schedule.",
    )
    parser.add_argument(
        "--save-loss-history",
        dest="save_loss_history",
        action="store_true",
        help="Save loss history npz/plots (default: enabled).",
    )
    parser.add_argument(
        "--no-save-loss-history",
        dest="save_loss_history",
        action="store_false",
        help="Disable saving loss history artifacts.",
    )

    # Optional preset
    parser.add_argument(
        "--preset",
        type=str,
        choices=["full", "quick"],
        default=None,
        help=(
            "Optional preset. 'quick' reduces epochs and images for fast runs; "
            "'full' uses defaults."
        ),
    )

    parser.set_defaults(save_loss_history=True, force_train=False)
    return parser.parse_args()


def maybe_apply_preset(args):
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
    print("\n--- [train_student_kd] Loading datasets ---")
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
        print(f"--- [train_student_kd] Using only {max_n} training images (Subset).")

    if args.max_val_images is not None:
        max_n = min(args.max_val_images, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(max_n)))
        print(f"--- [train_student_kd] Using only {max_n} validation images (Subset).")

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

    print(f"--- [train_student_kd] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def load_existing_checkpoint(model, candidate_paths, device, tag):
    """
    Try to load the first existing checkpoint from candidate_paths.
    Returns the path used, or None if nothing was loaded.
    """
    for ckpt_path in candidate_paths:
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            print(f"--- [{tag}] Found existing checkpoint at {ckpt_path}. Using it (warm start).")
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

    print("\n--- [train_student_kd] Configuration ---")
    if args.data_root:
        print(f"Data Root: {args.data_root}")
    print(f"Train Dir: {args.train_dir}")
    print(f"Val Dir:   {args.val_dir}")
    print(f"Patch Size: {args.patch_size}, Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}, Learning Rate: {args.learning_rate}")
    print(f"Device: {device}")
    print(f"Teacher Weights: {args.teacher_weights}")
    print(f"Checkpoint prefix: {args.checkpoint_prefix}")
    print(f"Drive ckpt dir:    {args.drive_checkpoint_dir}")
    print(f"Max train images:  {args.max_train_images}")
    print(f"Max val images:    {args.max_val_images}")
    print(f"Force train:       {args.force_train}")
    print("-----------------------------------\n")

    if not os.path.exists(args.teacher_weights):
        raise FileNotFoundError(
            f"Teacher weights not found at {args.teacher_weights}. "
            "Run train_teacher.py first or adjust --teacher-weights."
        )

    # File naming
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

    # Data
    train_loader, val_loader = build_dataloaders(args, device)

    print(f"--- [train_student_kd] Device: {device}")

    # Teacher (frozen)
    print(f"--- [train_student_kd] Loading teacher model from {args.teacher_weights}...")
    teacher = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(device)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("--- [train_student_kd] Teacher model loaded and frozen.")

    # Student
    print("--- [train_student_kd] Initializing student model...")
    student = UNetStudent(in_channels=4, out_channels=4).to(device)
    if not args.force_train:
        candidate_ckpts = [
            ckpt_final_drive,
            ckpt_final_local,
            ckpt_best_drive,
            ckpt_latest_drive,
            ckpt_latest_local,
        ]
        load_existing_checkpoint(student, candidate_ckpts, device, "train_student_kd")

    # Losses
    print("--- [train_student_kd] Initializing losses...")
    pixel_loss_fn     = CharbonnierLoss().to(device)
    feature_loss_fn   = nn.L1Loss().to(device)
    frequency_loss_fn = FFTAmpPhaseMultiScaleLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25,
        lambda_amp=1.0,
        lambda_phase=0.5,
        scales=(1.0, 0.5)
    ).to(device)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)

    # KD loss weights (same as your previous full KD)
    W_PIXEL   = 1.0
    W_FEATURE = 0.5
    W_FREQ    = 0.2
    W_LPIPS   = 0.1

    # Optimizer + scheduler + AMP
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    scaler = GradScaler()

    # Loss histories
    train_loss_history = []
    val_loss_history   = []
    best_val_loss      = float("inf")

    print("--- [train_student_kd] Starting Knowledge Distillation training...")

    for epoch in range(args.num_epochs):
        student.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]")
        for udc_batch, gt_batch in pbar:
            udc_batch = udc_batch.to(device, non_blocking=True)
            gt_batch  = gt_batch.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Teacher forward (no grad)
            with torch.no_grad():
                with autocast():
                    teacher_out_4ch, teacher_feat_spatial, _ = teacher(udc_batch)

            # Student forward
            with autocast():
                student_out_4ch, student_features_raw = student(udc_batch)

                # 1) Pixel loss
                loss_pixel = pixel_loss_fn(student_out_4ch, gt_batch)

                # 2) Spatial feature KD (resize student feature to teacher feature spatial size)
                student_features = F.interpolate(
                    student_features_raw,
                    size=teacher_feat_spatial.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                loss_feature_spatial = feature_loss_fn(student_features, teacher_feat_spatial)

                # 3) LPIPS on pseudo-RGB (first 3 channels)
                pred_rgb_slice = student_out_4ch[:, :3, :, :]
                gt_rgb_slice   = gt_batch[:, :3, :, :]
                loss_lpips = perceptual_loss_fn(
                    pred_rgb_slice * 2.0 - 1.0,
                    gt_rgb_slice   * 2.0 - 1.0
                ).mean()

            # 4) Frequency-based KD (float32)
            loss_freq_out = frequency_loss_fn(
                student_out_4ch.float(),
                teacher_out_4ch.float()
            )
            loss_feature_freq = frequency_loss_fn(
                student_features.float(),
                teacher_feat_spatial.float()
            )

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
        print(f"--- [train_student_kd] Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation (Charbonnier only, as before)
        student.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [val]")
            for udc_batch, gt_batch in pbar_val:
                udc_batch = udc_batch.to(device, non_blocking=True)
                gt_batch  = gt_batch.to(device,  non_blocking=True)

                with autocast():
                    student_out_4ch, _ = student(udc_batch)
                    loss_val = pixel_loss_fn(student_out_4ch, gt_batch)

                running_val_loss += loss_val.item()
                pbar_val.set_postfix({"val_loss": f"{loss_val.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"--- [train_student_kd] Epoch {epoch+1} Val Charbonnier Loss: {avg_val_loss:.4f}")

        scheduler.step()

        # Record histories
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Save latest checkpoints
        torch.save(student.state_dict(), ckpt_latest_local)
        torch.save(student.state_dict(), ckpt_latest_drive)
        print(f"Saved latest student checkpoint to {ckpt_latest_local} and {ckpt_latest_drive}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), ckpt_best_drive)
            print(f"New best val loss. Saved best student checkpoint to {ckpt_best_drive}")

        # Save loss history each epoch
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

    # Final named checkpoint
    torch.save(student.state_dict(), ckpt_final_local)
    torch.save(student.state_dict(), ckpt_final_drive)
    print(f"--- [train_student_kd] Final student saved as {ckpt_final_local} locally and on Drive")

    # Final loss curves plot
    if args.save_loss_history:
        epochs = np.arange(1, len(train_loss_history) + 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, train_loss_history, label="Train")
        ax.plot(epochs, val_loss_history,   label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (KD total: pixel + feat + freq + LPIPS)")
        ax.set_title("Student KD Training & Validation Loss")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_plot_local, dpi=150)
        plt.close(fig)

        # Copy plot to Drive
        try:
            import shutil
            shutil.copy(loss_plot_local, loss_plot_drive)
            print(f"Saved loss curves plot to {loss_plot_local} and {loss_plot_drive}")
        except Exception as e:
            print(f"Could not copy loss plot to Drive: {e}")


if __name__ == "__main__":
    main()
