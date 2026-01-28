import os
import sys

# Suppress verbose TensorFlow/CUDA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf

# Configure GPU Memory Growth (Prevent 'DNN library not found' due to pre-allocation)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Config Error: {e}")
import matplotlib.pyplot as plt
from datetime import datetime

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
GAIN = 128

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
# Matches training: alpha=0.1 creates a "fog" effect that the model learned.
# Larger kernel (61x61) needed because alpha=0.1 decays very slowly.
def create_exponential_kernel(size, alpha=0.1, gain=128):
    """ Generates an integer kernel with exponential decay: exp(-alpha * r) """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    r = np.sqrt(xx**2 + yy**2)
    kernel = np.exp(-alpha * r) 
    return (kernel * gain).astype(np.int16)

# Uses Alpha=0.1 to match training (creates "fog" effect)
KERNEL_BLUR = create_exponential_kernel(61, alpha=0.1, gain=GAIN)

# --- LABEL MAPPING ---
# HDF5 is Fixed Order (24 Classes). Model is 8-Class Subset (Alphabetical).
# We must map valid HDF5 indices to Model Indices (0-7).
#
# Model Class Order (Alphabetical 'processed/train' subset):
# 0: 16QAM
# 1: 32QAM
# 2: 4ASK
# 3: 64QAM
# 4: 8PSK
# 5: BPSK
# 6: OQPSK
# 7: QPSK

HDF5_TO_MODEL_MAP = {
    # HDF5 Index : Model Index
    1: 2,   # 4ASK  -> 4ASK
    3: 5,   # BPSK  -> BPSK
    4: 7,   # QPSK  -> QPSK
    5: 4,   # 8PSK  -> 8PSK
    12: 0,  # 16QAM -> 16QAM
    13: 1,  # 32QAM -> 32QAM
    14: 3,  # 64QAM -> 64QAM
    23: 6   # OQPSK -> OQPSK
}

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
    
    # DEBUG: Check for clipping
    print(f"DEBUG: u mapped range: [{u.min():.2f}, {u.max():.2f}] (Target: 0-{GRID_SIZE})")
    print(f"DEBUG: v mapped range: [{v.min():.2f}, {v.max():.2f}] (Target: 0-{GRID_SIZE})")

    # Quantize to integer indices
    u_idx = np.clip(np.round(u), 0, GRID_SIZE-1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, GRID_SIZE-1).astype(np.int16)
    
    # DEBUG: Check for clipping - Print only for first valid sample processed
    if not hasattr(hardware_gen_layer, "debug_printed"):
        print(f"DEBUG: Grid Mapping - u range: [{u.min():.2f}, {u.max():.2f}], v range: [{v.min():.2f}, {v.max():.2f}]")
        hardware_gen_layer.debug_printed = True

    # 2. ACCUMULATOR (The Bucket)
    # 16-bit memory initialized to 0
    accumulator = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32) # Increased to int32 to avoid overflow with GAIN
    
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
    # Ch3 (Blur) uses shift=5 because 61x61 kernel accumulates huge energy (Fog effect)
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, shift_val=2)
    ch2 = hardware_gen_layer(iq_samples, KERNEL_MEDIUM, shift_val=4)
    ch3 = hardware_gen_layer(iq_samples, KERNEL_BLUR, shift_val=5)
    
    # --- FOG FLOOR FIX ---
    # In training, every pixel receives tiny contributions from ALL 1024 IQ points
    # (global summation). Our kernel stamps only reach ~30px, leaving distant pixels black.
    # Add a "fog floor" to Channel 3 to simulate the ambient contribution.
    # The floor is proportional to the number of samples (more points = higher ambient).
    # Tunable parameter: FOG_FLOOR (0-255). Start with ~15 which is ~6% of max.
    FOG_FLOOR = 15
    ch3 = np.clip(ch3.astype(np.int16) + FOG_FLOOR, 0, 255).astype(np.uint8)
    
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
        
        # --- FILTERING LOGIC ---
        # 1. Unknown Class Hypothesis: The model only supports 8 Digital Mods.
        #    We must filter out OOK (0) and other analog/unsupported classes.
        #    Valid indices from HDF5 (based on comments above):
        #    4ASK(1), BPSK(3), QPSK(4), 8PSK(5), 16QAM(12), 32QAM(13), 64QAM(14), OQPSK(23)
        KNOWN_CLASSES_INDICES = [1, 3, 4, 5, 12, 13, 14, 23]
        
        # Get integer labels
        if Y.ndim > 1:
            y_integers = np.argmax(Y, axis=1)
        else:
            y_integers = Y.flatten()
            
        # Create masks
        class_mask = np.isin(y_integers, KNOWN_CLASSES_INDICES)
        snr_mask = (Z[:, 0] > 10) # High SNR
        
        # Combined Filter
        valid_indices = np.where(class_mask & snr_mask)[0]
        
        # Pick a high SNR sample (e.g., SNR > 10) to calibrate contrast clearly
        # Ideally filter where Z (SNR) > 10
        # high_snr_indices = np.where(Z[:, 0] > 10)[0] 
        # REPLACED by valid_indices
        
        if len(valid_indices) > 0:
            # SHUFFLE to ensure we test a mix of classes, not just the first one (4ASK)
            np.random.shuffle(valid_indices)
            
            samples_iq = X[valid_indices[:100]]
            high_snr_indices = valid_indices # Keep variable name for compatibility with validation block below
            print(f"Loaded {len(samples_iq)} real samples (SNR > 10, Valid Class).")
        else:
            # Fallback if no high SNR valid class found
            print("WARNING: No High SNR + Valid Class samples found. Searching for ANY Valid Class samples...")
            valid_any_snr = np.where(class_mask)[0]
            if len(valid_any_snr) > 0:
               samples_iq = X[valid_any_snr[:100]]
               high_snr_indices = valid_any_snr
               print(f"Loaded {len(samples_iq)} real samples (Valid Class, Any SNR).")
            else:
               raise ValueError("No samples found matching the 8 Target Modulations. Dataset might be incompatible.")
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
        # UPDATED SHIFTS based on previous runs:
        # Previous run showed Ch1 Max=255 at Shift=0. 
        # Ch3 needs much less shifting now that we have a larger kernel.
        # Let's try conservative low shifts.
        hw_batch = batch_process_hardware(samples_iq, shift_vals=(1, 3, 4))
        
        print(f"Running inference on {len(hw_batch)} samples...")
        
        # --- SAVE DEBUG IMAGES ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('results_local', 'debug_hardware', timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving first 5 generated images to {save_dir}...")
        for i in range(min(5, len(hw_batch))):
             filename = os.path.join(save_dir, f"hardware_sample_{i}.png")
             plt.imsave(filename, hw_batch[i])
             print(f"Saved {filename}")
        # -------------------------
        
        preds = model.predict(hw_batch)
        print("Inference successful. Check accuracy manually.")
        
        # Determine predicted classes vs Ground Truth if possible
        # (Assuming Y contains one-hot labels)
        if 'Y' in locals():
            # Handle One-Hot vs Sparse Integers
            if Y.ndim > 1 and Y.shape[1] > 1:
                # One-Hot Encoded
                full_y_true = np.argmax(Y, axis=1)
            else:
                # Likely Integers already
                full_y_true = Y.flatten()
            
            # Subset the labels
            y_raw_subset = full_y_true[high_snr_indices[:100]] if 'high_snr_indices' in locals() else full_y_true[:100]
            
            # Apply Mapping (HDF5 -> Model)
            y_true_subset = np.array([HDF5_TO_MODEL_MAP[y] for y in y_raw_subset])
            
            y_pred = np.argmax(preds, axis=1) # Model outputs are already in Model Space
            y_conf = np.max(preds, axis=1)
            
            # Map back to Names for readability
            MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
            
            # Print sample comparisions
            print("\nSample Predictions (True vs Pred):")
            for i in range(25): # Print more samples to catch variety
                true_name = MODEL_CLASS_NAMES[y_true_subset[i]]
                pred_name = MODEL_CLASS_NAMES[y_pred[i]]
                match = "MATCH" if y_true_subset[i] == y_pred[i] else "FAIL"
                print(f"  Sample {i}: True={true_name:<6} ({y_true_subset[i]}) | Pred={pred_name:<6} ({y_pred[i]}) | Conf={y_conf[i]:.2f} | {match}")

            acc = np.mean(y_true_subset == y_pred)
            print(f"\nValidation Accuracy on subset: {acc*100:.2f}%")
            
    else:
        print(f"\nModel not found at {model_path}. Skipping inference.")

if __name__ == "__main__":
    main()
