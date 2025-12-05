import os
import glob
import shutil
import argparse

def copy_subset(source_input_dir, source_gt_dir, target_input_dir, target_gt_dir, num_files):
    """Helper function to copy a subset of files."""

    os.makedirs(target_input_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(source_input_dir, "*.npy")))[:num_files]

    if not input_files:
        print(f"Error: No .npy files found in {source_input_dir}")
        return

    print(f"Copying {len(input_files)} files from {source_input_dir}...")

    for input_path in input_files:
        filename = os.path.basename(input_path)
        gt_path = os.path.join(source_gt_dir, filename)

        if os.path.exists(gt_path):
            shutil.copy(input_path, target_input_dir)
            shutil.copy(gt_path, target_gt_dir)
        else:
            print(f"Warning: Missing GT file for {filename}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create a small subset of the UDC dataset.")
    parser.add_argument("--source-root", type=str, default="/content/dataset/UDC-SIT", help="Root of the full dataset.")
    parser.add_argument("--target-root", type=str, default="data/UDC-SIT_subset", help="Where to create the subset.")
    parser.add_argument("--n-train", type=int, default=30, help="Number of training images.")
    parser.add_argument("--n-val", type=int, default=8, help="Number of validation images.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Creating subset from {args.source_root} to {args.target_root}...")
    
    # Training
    copy_subset(
        source_input_dir=os.path.join(args.source_root, "training/input"),
        source_gt_dir=os.path.join(args.source_root, "training/GT"),
        target_input_dir=os.path.join(args.target_root, "training/input"),
        target_gt_dir=os.path.join(args.target_root, "training/GT"),
        num_files=args.n_train
    )

    # Validation
    copy_subset(
        source_input_dir=os.path.join(args.source_root, "validation/input"),
        source_gt_dir=os.path.join(args.source_root, "validation/GT"),
        target_input_dir=os.path.join(args.target_root, "validation/input"),
        target_gt_dir=os.path.join(args.target_root, "validation/GT"),
        num_files=args.n_val
    )

    print("Subset creation complete.")

if __name__ == "__main__":
    main()
