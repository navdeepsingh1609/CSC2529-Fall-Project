# File: utils/visualizer.py
import os
import numpy as np
import rawpy as rp
import torch

# Store this in your project root in Colab
DEFAULT_DNG_PATH = 'background.dng' 

def postprocess_bayer_to_rgb(bayer_4ch_tensor, dng_path=DEFAULT_DNG_PATH):
    """
    Converts a 4-channel (GR, R, B, GB) tensor into a displayable
    3-channel (RGB) numpy image using rawpy.
    
    Args:
        bayer_4ch_tensor: A (1, 4, H, W) or (4, H, W) tensor 
                          in the [0, 1] range.
        dng_path: Path to the 'background.dng' file.

    Returns:
        rgb_image: (H, W, 3) float32 in [0, 1]
    """
    
    if not os.path.exists(dng_path):
        raise FileNotFoundError(
            f"Could not find '{dng_path}'. "
            "Please upload the 'background.dng' file from the UDC-SIT repo "
            "to your project's root directory in Colab."
        )

    # 1. Convert tensor to numpy and ensure shape (4, H, W)
    arr = bayer_4ch_tensor.squeeze().detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor after squeeze, got shape {arr.shape}")
    
    # If shape is (H, W, 4), move channels to first dim
    if arr.shape[-1] == 4:
        arr = np.moveaxis(arr, -1, 0)  # (4, H, W)

    if arr.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got shape {arr.shape}")

    # Scale [0,1] -> [0,1023] 10-bit integer values
    npy_4ch = (arr * 1023.0).astype(np.uint16)  # (4, H, W)
    
    # 2. Load the background .dng file as a template
    raw_template = rp.imread(dng_path)
    
    # 3. Insert into the Bayer pattern
    _, H, W = npy_4ch.shape
    h, w = H, W

    try:
        # raw_image is (H_sensor, W_sensor)
        raw_template.raw_image[0:h*2:2, 0:w*2:2] = npy_4ch[0] # GR
        raw_template.raw_image[0:h*2:2, 1:w*2:2] = npy_4ch[1] # R
        raw_template.raw_image[1:h*2:2, 0:w*2:2] = npy_4ch[2] # B
        raw_template.raw_image[1:h*2:2, 1:w*2:2] = npy_4ch[3] # GB
    except ValueError as e:
        print(f"Error: Shape mismatch during Bayer pattern insertion. "
              f"NPY shape {npy_4ch.shape}, Raw template shape {raw_template.raw_image.shape}")
        print("Falling back to simpler (but slower) insertion...")
        GR = raw_template.raw_image[0::2, 0::2]
        R  = raw_template.raw_image[0::2, 1::2]
        B  = raw_template.raw_image[1::2, 0::2]
        GB = raw_template.raw_image[1::2, 1::2]
        
        ph, pw = h, w  # Patch height/width
        
        GR[:ph, :pw] = npy_4ch[0]
        R[:ph, :pw]  = npy_4ch[1]
        B[:ph, :pw]  = npy_4ch[2]
        GB[:ph, :pw] = npy_4ch[3]

    # 4. Use rawpy's postprocess() to develop the RAW image
    rgb_image = raw_template.postprocess(
        use_camera_wb=True, 
        no_auto_bright=True,
        output_bps=8
    )  # (H_sensor, W_sensor, 3) uint8

    # 5. Crop to the valid patch area and normalize to [0,1]
    rgb_image = rgb_image[0:h, 0:w, :]
    return rgb_image.astype(np.float32) / 255.0
