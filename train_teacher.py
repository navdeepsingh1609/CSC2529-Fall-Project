# File: train_teacher.py
"""
Unified teacher training script for UDC-SIT.

- Trains FrequencyAwareTeacher (MambaIR + frequency block) on 4-ch .npy patches.
- Can run "full" or "quick" training just by changing CLI arguments:
    * --train-dir / --val-dir
    * --max-train-images / --max-val-images
    * --batch-size, --num-epochs

- Saves:
    * Latest checkpoint          -> <checkpoint_prefix>_latest.pth
    * Final checkpoint           -> <checkpoint_prefix>_final.pth
    * Loss history (npz)         -> <checkpoint_prefix>_loss_history.npz
    * Loss curves (PNG)          -> <checkpoint_prefix>_loss_curves.png
    * Mirrored into Drive folder -> --drive-checkpoint-dir
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt

# Speed vs reproducibility: favour speed on Colab
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Make sure external MambaIR is visible
project_root = os.getcwd()
mambair_path = os.path.join(project_root, 'models', 'external', 'MambaIR')
if mambair_path not in sys.path:
    sys.path.insert(0, mambair_path)

from datasets.udc_dataset import UDCDataset
from models.mambair_teacher import FrequencyAwareTeacher
from losses.frequency_loss import FFTAmpPhaseMultiScaleLoss
from losses.pixel_loss import CharbonnierLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FrequencyAwareTeacher on UDC-SIT (4-ch .npy patches)."
    )
    # Data
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
        help="Patch size used by UDCDataset and the teacher model.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="If set, limit number of training images (for quick experiments).",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="If set, limit number of validation images.",
    )

    # Training hyperparams
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=22,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
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
        default="teacher_4ch",
        help="Prefix for saved checkpoints and logs.",
    )
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/checkpoints",
        help="Google Drive directory where checkpoints & logs are mirrored.",
    )

    # Optional preset for convenience
    parser.add_argument(
        "--preset",
        type=str,
        choices=["full", "quick"],
        default=None,
        help=(
            "Optional convenience preset. "
            "'full' ~ default args, 'quick' uses small batch/epochs and few images. "
            "You can still override values explicitly."
        ),
    )

    return parser.parse_args()


def maybe_apply_preset(args):
    """Apply convenient defaults for --preset=quick/full without overriding explicit CLI."""
    if args.preset == "quick":
        # Only reduce if user left them at default-like values
        if args.batch_size == 8:
            args.batch_size = 2
        if args.num_epochs == 22:
            args.num_epochs = 5
        if args.max_train_images is None:
            args.max_train_images = 20
        if args.max_val_images is None:
            args.max_val_images = 10
    # 'full' just relies on explicit defaults
    return args


def build_dataloaders(args, device):
    """Create train/val DataLoaders with optional Subset limiting."""
    print("\n--- [train_teacher] Loading datasets ---")
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

    # Optional quick-mode limits
    if args.max_train_images is not None:
        max_n = min(args.max_train_images, len(train_dataset))
        indices = list(range(max_n))
        train_dataset = Subset(train_dataset, indices)
        print(f"--- [train_teacher] Using only {max_n} training images (Subset).")

    if args.max_val_images is not None:
        max_n = min(args.max_val_images, len(val_dataset))
        indices = list(range(max_n))
        val_dataset = Subset(val_dataset, indices)
        print(f"--- [train_teacher] Using only {max_n} validation images (Subset).")

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

    print(f"--- [train_teacher] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def main():
    args = parse_args()
    args = maybe_apply_preset(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.drive_checkpoint_dir, exist_ok=True)

    print("\n--- [train_teacher] Configuration ---")
    print(f"Train Dir: {args.train_dir}")
    print(f"Val Dir:   {args.val_dir}")
    print(f"Patch Size: {args.patch_size}, Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}, Learning Rate: {args.learning_rate}")
    print(f"Device: {device}")
    print(f"Checkpoint prefix: {args.checkpoint_prefix}")
    print(f"Drive ckpt dir:    {args.drive_checkpoint_dir}")
    print(f"Max train images:  {args.max_train_images}")
    print(f"Max val images:    {args.max_val_images}")
    print("-----------------------------------\n")

    # File naming based on prefix
    ckpt_latest_local = f"{args.checkpoint_prefix}_latest.pth"
    ckpt_final_local  = f"{args.checkpoint_prefix}_final.pth"
    ckpt_latest_drive = os.path.join(args.drive_checkpoint_dir, f"{args.checkpoint_prefix}_latest.pth")
    ckpt_best_drive   = os.path.join(args.drive_checkpoint_dir, f"{args.checkpoint_prefix}_best.pth")

    loss_hist_local = f"{args.checkpoint_prefix}_loss_history.npz"
    loss_hist_drive = os.path.join(args.drive_checkpoint_dir, f"{args.checkpoint_prefix}_loss_history.npz")
    loss_plot_local = f"{args.checkpoint_prefix}_loss_curves.png"
    loss_plot_drive = os.path.join(args.drive_checkpoint_dir, f"{args.checkpoint_prefix}_loss_curves.png")

    # Data
    train_loader, val_loader = build_dataloaders(args, device)

    # Model
    print("--- [train_teacher] Initializing model...")
    model = FrequencyAwareTeacher(
        in_channels=4,
        out_channels=4,
        img_size=args.patch_size
    ).to(device)

    # Losses
    print("--- [train_teacher] Initializing losses...")
    pixel_loss_fn = CharbonnierLoss().to(device)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    fft_loss_fn = FFTAmpPhaseMultiScaleLoss(
        loss_weight=1.0,
        focus_low_freq=True,
        cutoff=0.25,
        lambda_amp=1.0,
        lambda_phase=0.5,
        scales=(1.0, 0.5)
    ).to(device)

    # Loss weights (same as earlier)
    W_PIXEL = 1.0
    W_PERCEPTUAL = 0.1
    W_FFT = 0.05

    # Optimizer, scheduler, AMP
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    scaler = GradScaler()

    # Histories
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")

    print("--- [train_teacher] Starting training...")

    for epoch in range(args.num_epochs):
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]")
        for udc_batch, gt_batch in pbar:
            udc_batch = udc_batch.to(device, non_blocking=True)
            gt_batch  = gt_batch.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred_batch_4ch, _, _ = model(udc_batch)

                loss_pixel = pixel_loss_fn(pred_batch_4ch, gt_batch)

                # LPIPS on pseudo-RGB (first 3 channels)
                pred_rgb = pred_batch_4ch[:, :3, :, :]
                gt_rgb   = gt_batch[:, :3, :, :]
                loss_perceptual = perceptual_loss_fn(
                    pred_rgb * 2.0 - 1.0,
                    gt_rgb   * 2.0 - 1.0
                ).mean()

            loss_fft = fft_loss_fn(pred_batch_4ch.float(), gt_batch.float())

            total_loss = (W_PIXEL * loss_pixel) + \
                         (W_PERCEPTUAL * loss_perceptual) + \
                         (W_FFT * loss_fft)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += total_loss.item()
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"--- [train_teacher] Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [val]")
            for udc_batch, gt_batch in pbar_val:
                udc_batch = udc_batch.to(device, non_blocking=True)
                gt_batch  = gt_batch.to(device,  non_blocking=True)

                with autocast():
                    pred_batch_4ch, _, _ = model(udc_batch)
                    loss_val = pixel_loss_fn(pred_batch_4ch, gt_batch)

                running_val_loss += loss_val.item()
                pbar_val.set_postfix({"val_loss": f"{loss_val.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"--- [train_teacher] Epoch {epoch+1} Val Charbonnier Loss: {avg_val_loss:.4f}")

        scheduler.step()

        # Record histories
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Save latest checkpoint (local + drive)
        torch.save(model.state_dict(), ckpt_latest_local)
        torch.save(model.state_dict(), ckpt_latest_drive)
        print(f"Saved latest teacher checkpoint to {ckpt_latest_local} and {ckpt_latest_drive}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_best_drive)
            print(f"New best val loss. Saved best teacher checkpoint to {ckpt_best_drive}")

        # Save loss history every epoch
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
    torch.save(model.state_dict(), ckpt_final_local)
    torch.save(model.state_dict(), os.path.join(args.drive_checkpoint_dir, ckpt_final_local))
    print(f"--- [train_teacher] Final model saved as {ckpt_final_local} locally and on Drive")

    # Final loss curves plot
    epochs = np.arange(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss_history, label="Train")
    ax.plot(epochs, val_loss_history,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Charbonnier + FFT + LPIPS)")
    ax.set_title("Teacher Training & Validation Loss")
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
