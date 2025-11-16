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
    """
    
    if not os.path.exists(dng_path):
        raise FileNotFoundError(
            f"Could not find '{dng_path}'. "
            "Please upload the 'background.dng' file from the UDC-SIT repo "
            "to your project's root directory in Colab."
        )

    # 1. Convert tensor to numpy
    # (H, W, 4) in range [0, 1023]
    npy_4ch = bayer_4ch_tensor.squeeze().cpu().detach().numpy()
    npy_4ch = (npy_4ch * 1023.0).astype(np.uint16)
    
    # 2. Load the background .dng file as a template
    raw_template = rp.imread(dng_path)
    
    # 3. Get the dimensions and create an empty Bayer pattern
    # Note: raw_image is (H, W), but Bayer pattern is (H/2, W/2) for each channel
    h, w = npy_4ch.shape[1], npy_4ch.shape[2]
    raw_h, raw_w = raw_template.raw_image.shape
    
    # 4. Manually insert the 4 channels into the Bayer pattern
    # This copies the structure from the official visualize_sit.py
    try:
        raw_template.raw_image[0:h*2:2, 0:w*2:2] = npy_4ch[0] # GR
        raw_template.raw_image[0:h*2:2, 1:w*2:2] = npy_4ch[1] # R
        raw_template.raw_image[1:h*2:2, 0:w*2:2] = npy_4ch[2] # B
        raw_template.raw_image[1:h*2:2, 1:w*2:2] = npy_4ch[3] # GB
    except ValueError as e:
        print(f"Error: Shape mismatch during Bayer pattern insertion. "
              f"NPY shape {npy_4ch.shape}, Raw template shape {raw_template.raw_image.shape}")
        print("Falling back to simpler (but slower) insertion...")
        # Fallback for mismatched shapes (e.g., 256x256 patch)
        GR = raw_template.raw_image[0::2, 0::2]
        R  = raw_template.raw_image[0::2, 1::2]
        B  = raw_template.raw_image[1::2, 0::2]
        GB = raw_template.raw_image[1::2, 1::2]
        
        ph, pw = npy_4ch.shape[1], npy_4ch.shape[2] # Patch height/width
        
        GR[:ph, :pw] = npy_4ch[0]
        R[:ph, :pw]  = npy_4ch[1]
        B[:ph, :pw]  = npy_4ch[2]
        GB[:ph, :pw] = npy_4ch[3]

    # 5. Use rawpy's postprocess() to develop the RAW image
    # This does demosaicing, white balancing, color correction, etc.
    # It returns a (H, W, 3) RGB image in range [0, 255]
    rgb_image = raw_template.postprocess(
        use_camera_wb=True, 
        no_auto_bright=True,
        output_bps=8
    )

    # 6. Crop to the valid image area (from official script)
    # and normalize to [0, 1] for display/metrics
    rgb_image = rgb_image[0:h, 0:w, :]
    return rgb_image.astype(np.float32) / 255.0