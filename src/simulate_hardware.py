import numpy as np
import tensorflow as tf
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing tools to load data and model
# Adjust paths if your model/data are elsewhere
from common.squeezenet import build_squeezenet_v11
from common.data_loader import load_data_sample

# --- HARDWARE PARAMETERS ---
GRID_SIZE = 224

# SCALE FACTOR: We scale up the kernels to utilize the 16-bit accumulator (0-65535)
# This acts as our "Fixed Point Gain"
GAIN = 32 

# Kernel definitions (The "Stamp")
# 1. Sharp (Corresponds to alpha=10)
KERNEL_SHARP = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.int16) * GAIN
# 2. Medium (Corresponds to alpha=1.0)
KERNEL_MEDIUM = np.array([[1, 2, 1],
                          [2, 8, 2],
                          [1, 2, 1]], dtype=np.int16) * GAIN
# 3. Blur (Corresponds to alpha=0.1)
KERNEL_BLUR = np.ones((5, 5), dtype=np.int16) * GAIN

def hardware_gen_layer(iq_samples, kernel, shift_val=4):
    """
    The Digital Twin of the FPGA Pipeline.
    Strictly integer logic. No floats allowed after coordinate mapping.
    """
    # 1. COORDINATE MAPPING (Float -> Int)
    # Map [-3.5, 3.5] to [0, 224]. 
    # Logic: (val + 3.5) * (224 / 7) = (val + 3.5) * 32
    scale = GRID_SIZE / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale
    
    # Quantize to integer indices
    u_idx = np.clip(np.round(u), 0, GRID_SIZE-1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, GRID_SIZE-1).astype(np.int16)

    # 2. ACCUMULATOR (The Bucket)
    # 16-bit memory initialized to 0
    accumulator = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    
    # 3. STAMPING (The Scatter)
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    for x, y in zip(u_idx, v_idx):
        # Calculate bounds to stay within grid
        x_min = max(0, x - pad_h)
        x_max = min(GRID_SIZE, x + pad_h + 1)
        y_min = max(0, y - pad_w)
        y_max = min(GRID_SIZE, y + pad_w + 1)
        
        # Calculate overlap slice on kernel
        k_x_min = pad_h - (x - x_min)
        k_x_max = k_x_min + (x_max - x_min)
        k_y_min = pad_w - (y - y_min)
        k_y_max = k_y_min + (y_max - y_min)

        # "Stamp" the kernel (Accumulate)
        accumulator[x_min:x_max, y_min:y_max] += kernel[k_x_min:k_x_max, k_y_min:k_y_max]

    # 4. FUNNEL (Shift & Clip)
    # Bit-shift division
    output = accumulator >> shift_val
    # Saturate at 255 (UINT8 limit)
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

def build_hardware_image(iq_samples):
    """Generates the full 3-channel image using hardware logic."""
    # We use different shift values because wider kernels accumulate more mass
    # These shift values must be tuned!
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, shift_val=2)
    ch2 = hardware_gen_layer(iq_samples, KERNEL_MEDIUM, shift_val=4)
    ch3 = hardware_gen_layer(iq_samples, KERNEL_BLUR, shift_val=5)
    
    # Stack to create (224, 224, 3)
    return np.stack([ch1, ch2, ch3], axis=-1)

def batch_process_hardware(iq_batch, shift_vals):
    """
    Helps process a whole batch of samples (looping sequentially like hardware pipeline).
    shift_vals: tuple of 3 ints (shift_ch1, shift_ch2, shift_ch3)
    """
    batch_out = []
    s1, s2, s3 = shift_vals
    
    for i in range(len(iq_batch)):
        ch1 = hardware_gen_layer(iq_batch[i], KERNEL_SHARP, shift_val=s1)
        ch2 = hardware_gen_layer(iq_batch[i], KERNEL_MEDIUM, shift_val=s2)
        ch3 = hardware_gen_layer(iq_batch[i], KERNEL_BLUR, shift_val=s3)
        img = np.stack([ch1, ch2, ch3], axis=-1)
        batch_out.append(img)
        
    return np.array(batch_out)

def main():
    print("--- Digital Twin Hardware Simulation ---")
    
    # 1. Load Data (REAL CLUSTER DATA)
    # Using the path found in 'evaluate.py'
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    # Use the SqueezeNet model as this is the target of the project (based on file listing)
    model_path = 'models/squeezenet_v11_rmsprop.h5'

    # Try to load real data, fall back to synthetic if on local machine without data
    try:
        print(f"Attempting to load data from {data_path}...")
        X, Y, Z = load_data_sample(data_path)
        # Pick a high SNR sample (e.g., SNR > 10) to calibrate contrast clearly
        # Ideally filter where Z (SNR) > 10
        high_snr_indices = np.where(Z[:, 0] > 10)[0]
        if len(high_snr_indices) > 0:
            samples_iq = X[high_snr_indices[:100]]
            print(f"Loaded {len(samples_iq)} real samples (SNR > 10).")
        else:
            samples_iq = X[0:100] 
            print(f"Loaded {len(samples_iq)} real samples (Mixed SNR).")
    except Exception as e:
        print(f"Could not load real data ({e}). Using synthetic QPSK for test.")
        # Synthetic QPSK (4 clusters)
        centers = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        indices = np.random.randint(0, 4, 102400)
        chosen = centers[indices]
        noise = np.random.normal(0, 0.3, (102400, 2))
        samples_iq = (chosen + noise).reshape(100, 1024, 2)

    # 2. Run Shift Sweep (Calibration)
    # We need to find the shift that puts the Mean pixel value around 20-50
    # and the Max around 255 (without saturating the whole image).
    print("\n--- Running Shift Sweep (Calibration) ---")
    print("NOTE: Kernels have been scaled by x32 (GAIN) to fix low contrast.")
    
    kernels = [
        ('Sharp (Ch1, Alpha=10)', KERNEL_SHARP), 
        ('Medium (Ch2, Alpha=1)', KERNEL_MEDIUM), 
        ('Blur (Ch3, Alpha=0.1)', KERNEL_BLUR)
    ]

    for name, kernel in kernels:
        print(f"\nEvaluating {name}:")
        for shift in range(12): # Expanded range due to GAIN
            # Process just the first sample for quick stats
            img = hardware_gen_layer(samples_iq[0], kernel, shift_val=shift)
            print(f"  Shift={shift}: Min={img.min():3d}, Max={img.max():3d}, Mean={img.mean():6.2f}")

    # 3. Model Verification
    if os.path.exists(model_path):
        print(f"\nLoading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # NOTE: You must update these SHIFT values based on the sweep output above!
        # Initial Guess with GAIN=32:
        # If Gain=32 (shift 5), and we needed shift 4 before, maybe shift 8-10 now?
        # Let's try conservative shifts to avoid black images. 
        # You will tune these after seeing the log output.
        print("Running inference with estimated shifts (Check logs if images are bad)...")
        hw_batch = batch_process_hardware(samples_iq, shift_vals=(5, 6, 8))
        
        print(f"Running inference on {len(hw_batch)} samples...")
        preds = model.predict(hw_batch)
        print("Inference successful. Check accuracy manually.")
        
        # Determine predicted classes vs Ground Truth if possible
        # (Assuming Y contains one-hot labels)
        if 'Y' in locals():
            y_true = np.argmax(Y[high_snr_indices[:100]] if 'high_snr_indices' in locals() else Y[:100], axis=1)
            y_pred = np.argmax(preds, axis=1)
            acc = np.mean(y_true == y_pred)
            print(f"Validation Accuracy on subset: {acc*100:.2f}%")
            
    else:
        print(f"\nModel not found at {model_path}. Skipping inference.")

if __name__ == "__main__":
    main()
