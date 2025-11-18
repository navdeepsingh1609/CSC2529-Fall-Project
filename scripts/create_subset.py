import os
import glob
import shutil

# --- Config ---
# --- [FIX] Point SOURCE_DIR to the extracted dataset location ---
SOURCE_DIR = "/content/dataset/UDC-SIT" 
# --- [END FIX] ---

# Target paths (where the small subset will be)
TARGET_DIR = "data/UDC-SIT_subset"

NUM_TRAIN_FILES = 10  # Number of training files to copy
NUM_VAL_FILES = 10     # Number of validation files to copy
# --------------

def copy_subset(source_input_dir, source_gt_dir, target_input_dir, target_gt_dir, num_files):
    """Helper function to copy a subset of files."""
    
    # Create target directories
    os.makedirs(target_input_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)
    
    # Get a list of input files
    input_files = sorted(glob.glob(os.path.join(source_input_dir, "*.npy")))[:num_files]
    
    if not input_files:
        print(f"Error: No .npy files found in {source_input_dir}")
        return

    print(f"Copying {len(input_files)} files from {source_input_dir}...")

    for input_path in input_files:
        filename = os.path.basename(input_path)
        gt_path = os.path.join(source_gt_dir, filename)
        
        # Check if corresponding GT file exists
        if os.path.exists(gt_path):
            # Copy input file
            shutil.copy(input_path, target_input_dir)
            # Copy GT file
            shutil.copy(gt_path, target_gt_dir)
        else:
            print(f"Warning: Missing GT file for {filename}")

def main():
    print("Creating training subset...")
    copy_subset(
        source_input_dir=os.path.join(SOURCE_DIR, "training/input"),
        source_gt_dir=os.path.join(SOURCE_DIR, "training/GT"),
        target_input_dir=os.path.join(TARGET_DIR, "train/input"),
        target_gt_dir=os.path.join(TARGET_DIR, "train/GT"),
        num_files=NUM_TRAIN_FILES
    )
    
    print("Creating validation subset...")
    copy_subset(
        source_input_dir=os.path.join(SOURCE_DIR, "validation/input"),
        source_gt_dir=os.path.join(SOURCE_DIR, "validation/GT"),
        target_input_dir=os.path.join(TARGET_DIR, "val/input"),
        target_gt_dir=os.path.join(TARGET_DIR, "val/GT"),
        num_files=NUM_VAL_FILES
    )
    
    print("Subset creation complete.")

if __name__ == "__main__":
    main()