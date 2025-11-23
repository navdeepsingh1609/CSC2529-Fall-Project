# File: testing_udc.py
import os
import sys
import argparse
import csv
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torch.cuda import is_available as cuda_is_available
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Add external MambaIR path if needed (assuming this file lives at repo root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAMBAIR_PATH = os.path.join(PROJECT_ROOT, "models", "external", "MambaIR")
if MAMBAIR_PATH not in sys.path:
    sys.path.insert(0, MAMBAIR_PATH)

from models.mambair_teacher import FrequencyAwareTeacher
from models.unet_student import UNetStudent


def bayer4_to_rgb_balanced(arr_4ch, r_gain: float = 1.9, b_gain: float = 1.9):
    """
    Balanced Bayer-to-RGB view used for visualization (and LPIPS in testing_full.py).

    Accepts (H,W,4) or (4,H,W) in [0,1] or [0,1023] and returns (H,W,3) in [0,1].
    """
    arr = np.asarray(arr_4ch, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    if arr.shape[0] == 4:
        GR, R, B, GB = arr
    elif arr.shape[-1] == 4:
        GR = arr[..., 0]
        R = arr[..., 1]
        B = arr[..., 2]
        GB = arr[..., 3]
    else:
        raise ValueError(f"Expected channel dim of 4, got shape {arr.shape}")

    scale = 1023.0 if arr.max() > 2.0 else 1.0
    G = (GR + GB) / (2.0 * scale)
    Rn = (R / scale) * r_gain
    Bn = (B / scale) * b_gain
    rgb = np.stack([Rn, G, Bn], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def load_full_npy_pair(input_path: str, gt_path: str):
    """
    Load a full-resolution 4-channel UDC-SIT pair from .npy files.

    Both input and GT are expected to be shaped (H, W, 4) with values in [0, 1023].
    Returns:
        udc_4ch: torch.Tensor (4, H, W) in [0,1]
        gt_4ch:  torch.Tensor (4, H, W) in [0,1]
    """
    udc = np.load(input_path)  # (H, W, 4)
    gt = np.load(gt_path)
    if udc.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {udc.shape} vs {gt.shape} for {input_path} and {gt_path}")

    udc_t = torch.from_numpy(udc).permute(2, 0, 1).float() / 1023.0
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).float() / 1023.0
    return udc_t, gt_t


def pad_to_multiple(tensor: torch.Tensor, patch_size: int):
    """
    Reflect-pad a CHW tensor so that H and W become multiples of patch_size.

    Returns:
        padded: (C, H_pad, W_pad)
        H_orig, W_orig: original spatial sizes
    """
    _, H, W = tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return tensor, H, W

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, H, W


def run_model_tiled(
    model: torch.nn.Module,
    udc_full_4ch: torch.Tensor,
    patch_size: int,
    patch_batch: int,
    device: torch.device,
):
    """
    Run model on full 4-ch image by tiling into non-overlapping patches.

    Args:
        model: restoration network, expects (B,4,H,W) in [0,1].
        udc_full_4ch: (4, H, W) tensor in [0,1].
        patch_size: patch height/width.
        patch_batch: number of patches per forward pass.
        device: torch.device.

    Returns:
        pred_full_4ch: (4, H, W) tensor in [0,1].
    """
    model.eval()
    with torch.no_grad():
        x, H_orig, W_orig = pad_to_multiple(udc_full_4ch, patch_size)  # (4, H_pad, W_pad)
        C, H_pad, W_pad = x.shape

        x = x.unsqueeze(0)  # (1, 4, H_pad, W_pad)

        ys = list(range(0, H_pad, patch_size))
        xs = list(range(0, W_pad, patch_size))
        coords = [(y, x_) for y in ys for x_ in xs]

        pred_full = torch.zeros_like(x)

        for i in range(0, len(coords), patch_batch):
            batch_coords = coords[i : i + patch_batch]
            patches = []
            for (y, x_) in batch_coords:
                patch = x[:, :, y : y + patch_size, x_ : x_ + patch_size]  # (1,4,ps,ps)
                patches.append(patch)
            patches = torch.cat(patches, dim=0).to(device)  # (B,4,ps,ps)

            with amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                out, *_ = model(patches)  # (B,4,ps,ps)

            out = out.detach().cpu()

            for b, (y, x_) in enumerate(batch_coords):
                pred_full[:, :, y : y + patch_size, x_ : x_ + patch_size] = out[b : b + 1]

        pred_full = pred_full[:, :, :H_orig, :W_orig]
        return pred_full.squeeze(0)  # (4, H_orig, W_orig)


def run_model_full_image(
    model: torch.nn.Module,
    udc_full_4ch: torch.Tensor,
    device: torch.device,
    use_tiling: bool,
    patch_size: int,
    patch_batch: int,
):
    """
    Try full-image inference; if that fails with CUDA OOM, fall back to tiled inference.

    Args:
        model: restoration network
        udc_full_4ch: (4, H, W) tensor in [0,1]
        device: cuda or cpu
        use_tiling: if True, skip full-image and directly use tiling
    """
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not use_tiling:
        x = udc_full_4ch.unsqueeze(0).to(device)  # (1,4,H,W)
        try:
            with torch.no_grad(), amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                out, *_ = model(x)  # (1,4,H,W)
            return out.detach().cpu().squeeze(0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                print("[run_model_full_image] CUDA OOM on full image, falling back to tiled inference...")
                torch.cuda.empty_cache()
            else:
                raise

    # Fallback or forced tiling
    return run_model_tiled(model, udc_full_4ch, patch_size, patch_batch, device)


def compute_metrics_raw_and_lpips(
    pred_4ch: torch.Tensor,
    gt_4ch: torch.Tensor,
    lpips_model: lpips.LPIPS,
    device: torch.device,
):
    """
    Compute PSNR/SSIM in 4-channel (Bayer-like) domain and LPIPS on pseudo-RGB.

    Args:
        pred_4ch, gt_4ch: (4,H,W) in [0,1]
        lpips_model: LPIPS(net='vgg') already moved to device.

    Returns:
        psnr (float), ssim (float), lpips_rgb (float)
    """
    pred = pred_4ch.clamp(0.0, 1.0)
    gt = gt_4ch.clamp(0.0, 1.0)

    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    gt_np = gt.permute(1, 2, 0).cpu().numpy()

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    # LPIPS on pseudo-RGB (channels 0,1,2)
    pred_rgb = pred[:3, :, :].unsqueeze(0).to(device)  # (1,3,H,W)
    gt_rgb = gt[:3, :, :].unsqueeze(0).to(device)

    pred_rgb_lp = pred_rgb * 2.0 - 1.0
    gt_rgb_lp = gt_rgb * 2.0 - 1.0

    with torch.no_grad():
        lpv = lpips_model(pred_rgb_lp, gt_rgb_lp).item()

    return psnr, ssim, lpv


def evaluate_model_on_split(
    model_name: str,
    model_type: str,
    weights_path: str,
    data_root: str,
    split: str,
    patch_size: int,
    patch_batch: int,
    max_images: int,
    results_root: str,
    drive_results_root: str,
    save_npy: bool,
    use_tiling: bool,
    eval_mode: str,
    num_plot_examples: int,
    lpips_model: lpips.LPIPS,
    device: torch.device,
):
    """
    Evaluate a teacher or student model on the given split.

    Saves:
        - 4-ch predictions as .npy in [0, 1023] if save_npy=True
        - metrics_raw.csv and metrics_summary.txt
        - copies same to Drive (including npy dir) if drive_results_root is provided
        - optional visualization panels (balanced Bayer->RGB) for first num_plot_examples
    Args:
        eval_mode: "full" uses full-image/tiled inference; "center_patch" runs only on a center crop of size patch_size.
    """
    input_dir = os.path.join(data_root, split, "input")
    gt_dir = os.path.join(data_root, split, "GT")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input dir: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Missing GT dir: {gt_dir}")

    input_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    if max_images is not None:
        input_files = input_files[:max_images]

    num_images = len(input_files)
    print(f"Data root: {data_root}")
    print(f"Num images: {num_images}")
    print(f"Eval mode: {eval_mode}")

    # Build model
    if model_type == "teacher":
        print(f"--- [Teacher] Initializing with 4 in-channels and 4 out-channels.")
        model = FrequencyAwareTeacher(in_channels=4, out_channels=4).to(device)
        print(f"--- [Teacher] Model initialized (MambaIR + Freq residual gating).")
    elif model_type == "student":
        print(f"--- [Student] Initializing with 4 in-channels and 4 out-channels.")
        model = UNetStudent(in_channels=4, out_channels=4).to(device)
        print(f"--- [Student] Model initialized with frequency blocks.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Output dirs
    model_result_dir = os.path.join(results_root, f"{model_name}_{split}")
    os.makedirs(model_result_dir, exist_ok=True)

    npy_out_dir = None
    if save_npy:
        npy_out_dir = os.path.join(model_result_dir, "npy")
        os.makedirs(npy_out_dir, exist_ok=True)

    metrics_csv_path = os.path.join(model_result_dir, "metrics_raw.csv")
    summary_path = os.path.join(model_result_dir, "metrics_summary.txt")

    rows = []
    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []

    # For visualization panels
    plot_inputs, plot_preds, plot_gts, plot_ids = [], [], [], []

    pbar = tqdm(input_files, desc=f"[{model_name}] images", unit="img")
    for inp_path in pbar:
        base = os.path.basename(inp_path)
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            print(f"[{model_name}] WARNING: GT missing for {base}, skipping.")
            continue

        # --- Data loading based on eval_mode ---
        if eval_mode == "center_patch":
            arr_inp = np.load(inp_path).astype(np.float32)
            arr_gt = np.load(gt_path).astype(np.float32)
            H, W, _ = arr_inp.shape
            ps = patch_size
            if ps > H or ps > W:
                raise ValueError(f"Patch size {ps} exceeds image size {(H, W)} for {base}")
            r0 = (H - ps) // 2
            c0 = (W - ps) // 2
            r1, c1 = r0 + ps, c0 + ps
            inp_patch = arr_inp[r0:r1, c0:c1, :]
            gt_patch = arr_gt[r0:r1, c0:c1, :]

            udc_full = torch.from_numpy(inp_patch).permute(2, 0, 1).float() / 1023.0
            gt_full = torch.from_numpy(gt_patch).permute(2, 0, 1).float() / 1023.0
            udc_full = udc_full.to(device)
            gt_full = gt_full.to(device)

            start_time = time.perf_counter()
            with torch.no_grad(), amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                pred_full, *_ = model(udc_full.unsqueeze(0))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            pred_full = pred_full.squeeze(0).cpu()
            udc_full = udc_full.cpu()
            gt_full = gt_full.cpu()
        else:
            udc_full, gt_full = load_full_npy_pair(inp_path, gt_path)
            udc_full = udc_full.to(device)

            # Inference with timing (sync on CUDA for accurate measurement)
            start_time = time.perf_counter()
            pred_full = run_model_full_image(
                model=model,
                udc_full_4ch=udc_full,
                device=device,
                use_tiling=use_tiling,
                patch_size=patch_size,
                patch_batch=patch_batch,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            pred_full = pred_full.cpu()
            udc_full = udc_full.cpu()
            gt_full = gt_full.cpu()

        # Save predictions as 4-ch .npy in [0,1023]
        if save_npy and npy_out_dir is not None:
            out_npy_path = os.path.join(npy_out_dir, base)
            pred_np = pred_full.clamp(0.0, 1.0).permute(1, 2, 0).numpy().astype(np.float32) * 1023.0
            np.save(out_npy_path, pred_np)

        # Metrics
        psnr, ssim, lpv = compute_metrics_raw_and_lpips(pred_full, gt_full, lpips_model, device)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpv)
        time_list.append(elapsed)

        rows.append(
            {
                "filename": base,
                "psnr_bayer": psnr,
                "ssim_bayer": ssim,
                "lpips_rgb": lpv,
                "time_ms": elapsed * 1000.0,
            }
        )

        pbar.set_postfix(
            {
                "PSNR": f"{psnr:.2f}",
                "SSIM": f"{ssim:.3f}",
                "LPIPS": f"{lpv:.3f}",
                "time_ms": f"{elapsed*1000.0:.1f}",
            }
        )

        # Collect a few samples for visualization using balanced Bayer->RGB
        if len(plot_inputs) < num_plot_examples:
            plot_ids.append(base)
            plot_inputs.append(bayer4_to_rgb_balanced(udc_full.cpu().numpy()))
            plot_preds.append(bayer4_to_rgb_balanced(pred_full.cpu().numpy()))
            plot_gts.append(bayer4_to_rgb_balanced(gt_full.cpu().numpy()))

    # Save CSV
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "psnr_bayer", "ssim_bayer", "lpips_rgb", "time_ms"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else float("nan")
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")
    mean_time_ms = float(np.mean(time_list) * 1000.0) if time_list else float("nan")

    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Type:  {model_type}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Num images: {len(psnr_list)}\n\n")
        f.write(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB\n")
        f.write(f"Mean SSIM (Bayer): {mean_ssim:.4f}\n")
        f.write(f"Mean LPIPS (RGB):  {mean_lpips:.4f}\n")
        f.write(f"Mean Inference Time: {mean_time_ms:.3f} ms\n")

    print(f"\n--- [testing_udc] {model_name} results on '{split}':")
    print(f"Mean PSNR (Bayer): {mean_psnr:.4f} dB")
    print(f"Mean SSIM (Bayer): {mean_ssim:.4f}")
    print(f"Mean LPIPS (RGB):  {mean_lpips:.4f}")
    print(f"Mean Inference:    {mean_time_ms:.3f} ms")
    print(f"Saved CSV to {metrics_csv_path}")
    print(f"Saved summary to {summary_path}")

    # Visualization panels using balanced Bayer->RGB
    vis_path = None
    if plot_inputs:
        rows = len(plot_inputs)
        cols = 3  # Input, Prediction, GT
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(rows):
            axes[i, 0].imshow(plot_inputs[i])
            axes[i, 0].set_title(f"{plot_ids[i]} - Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(plot_preds[i])
            axes[i, 1].set_title("Prediction")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(plot_gts[i])
            axes[i, 2].set_title("GT")
            axes[i, 2].axis("off")

        plt.tight_layout()
        vis_path = os.path.join(model_result_dir, f"{model_name}_{split}_visualization.png")
        plt.savefig(vis_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization plot to {vis_path}")

    # Copy metrics + npy predictions to Drive
    if drive_results_root is not None and len(drive_results_root) > 0:
        import shutil

        drive_model_dir = os.path.join(drive_results_root, f"{model_name}_{split}")
        os.makedirs(drive_model_dir, exist_ok=True)

        shutil.copy(metrics_csv_path, os.path.join(drive_model_dir, "metrics_raw.csv"))
        shutil.copy(summary_path, os.path.join(drive_model_dir, "metrics_summary.txt"))
        if vis_path is not None:
            shutil.copy(vis_path, os.path.join(drive_model_dir, os.path.basename(vis_path)))

        if save_npy and npy_out_dir is not None and os.path.isdir(npy_out_dir):
            drive_npy_dir = os.path.join(drive_model_dir, "npy")
            shutil.copytree(npy_out_dir, drive_npy_dir, dirs_exist_ok=True)

        print(f"Copied results to Drive: {drive_model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="UDC-SIT testing script (raw-domain metrics + 4-ch predictions, teacher + student)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/dataset/UDC-SIT/UDC-SIT",
        help="Root of UDC-SIT dataset containing train/validation/testing folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "testing"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size for tiled inference or center-patch eval (see --eval-mode).",
    )
    parser.add_argument(
        "--patch-batch",
        type=int,
        default=8,
        help="Number of patches per forward pass during tiled inference.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="If set, limit the number of images evaluated.",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="teacher_4ch_22epochs_bs8.pth",
        help="Path to teacher checkpoint (.pth).",
    )
    parser.add_argument(
        "--student-ckpt",
        type=str,
        default="student_distilled_4ch_full_data.pth",
        help="Path to student checkpoint (.pth).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/Results/Model2",
        help="Local directory to store metrics and predictions.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="run1",
        help="Subfolder name created under both results roots for this run.",
    )
    parser.add_argument(
        "--drive-results-root",
        type=str,
        default="/content/drive/MyDrive/Computational Imaging Project/Results/Model2",
        help="Google Drive directory to mirror metrics and NPY predictions. Leave empty to disable.",
    )
    parser.add_argument(
        "--no-save-npy",
        action="store_true",
        help="Disable saving 4-ch prediction .npy files.",
    )
    parser.add_argument(
        "--num-plot-examples",
        type=int,
        default=0,
        help="Number of examples to visualize per model/split (balanced Bayer->RGB).",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="full",
        choices=["full", "center_patch"],
        help="full: run on entire image (full or tiled). center_patch: evaluate only the center patch of size --patch-size.",
    )
    parser.add_argument(
        "--use-tiling",
        action="store_true",
        help="Force tiled inference even when full-image inference might be possible.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if cuda_is_available() else "cpu",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Compose final output roots (local + Drive) with optional run subfolder
    results_root = os.path.join(args.results_root, args.results_name) if args.results_name else args.results_root
    drive_results_root = (
        os.path.join(args.drive_results_root, args.results_name)
        if args.drive_results_root is not None and len(args.drive_results_root) > 0 and args.results_name
        else args.drive_results_root
    )

    os.makedirs(results_root, exist_ok=True)
    if drive_results_root is not None and len(str(drive_results_root)) > 0:
        os.makedirs(drive_results_root, exist_ok=True)

    print("\n--- [testing_udc] Configuration ---")
    print(f"Data root:      {args.data_root}")
    print(f"Split:          {args.split}")
    print(f"Patch size:     {args.patch_size}")
    print(f"Patch batch:    {args.patch_batch}")
    print(f"Max images:     {args.max_images}")
    print(f"Teacher ckpt:   {args.teacher_ckpt}")
    print(f"Student ckpt:   {args.student_ckpt}")
    print(f"Results root:   {results_root}")
    print(f"Drive results:  {drive_results_root}")
    print(f"Plot examples:  {args.num_plot_examples}")
    print(f"Eval mode:      {args.eval_mode}")
    print(f"Use tiling:     {args.use_tiling}")
    print(f"Save NPY:       {not args.no_save_npy}")
    print(f"Device:         {device.type}")
    print("-----------------------------------\n")

    # LPIPS
    print("Setting up LPIPS (VGG) on device:", device.type)
    lpips_model = lpips.LPIPS(net="vgg").to(device)

    # Build model configs
    model_cfgs = []
    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        model_cfgs.append(
            {"name": "teacher_full", "model_type": "teacher", "weights": args.teacher_ckpt}
        )
    else:
        print("[testing_udc] Teacher checkpoint not found, skipping teacher.")

    if args.student_ckpt and os.path.exists(args.student_ckpt):
        model_cfgs.append(
            {"name": "student_full", "model_type": "student", "weights": args.student_ckpt}
        )
    else:
        print("[testing_udc] Student checkpoint not found, skipping student.")

    if not model_cfgs:
        raise RuntimeError("No valid checkpoints found for teacher or student.")

    for cfg in model_cfgs:
        print(
            f"\n=== [testing_udc] Evaluating {cfg['name']} ({cfg['model_type']}) on split '{args.split}' ==="
        )
        evaluate_model_on_split(
            model_name=cfg["name"],
            model_type=cfg["model_type"],
            weights_path=cfg["weights"],
            data_root=args.data_root,
            split=args.split,
            patch_size=args.patch_size,
            patch_batch=args.patch_batch,
            max_images=args.max_images,
            results_root=results_root,
            drive_results_root=drive_results_root,
            save_npy=not args.no_save_npy,
            use_tiling=args.use_tiling,
            eval_mode=args.eval_mode,
            num_plot_examples=args.num_plot_examples,
            lpips_model=lpips_model,
            device=device,
        )


if __name__ == "__main__":
    main()
