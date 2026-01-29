"""
Hardware Image Generation - Digital Twin for FPGA AMC Pipeline
PROVEN WORKING CONFIG from git history (Version 1 - produces great images)

Key principles:
1. Fixed GAIN, fixed bit-shifts = FPGA-streamable (no dynamic max)
2. Integer-only math after coordinate mapping
3. 3x3 kernels for Ch1/Ch2 = minimal FPGA footprint
4. 11x11 uniform for Ch3 = no coefficient LUT needed
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.squeezenet import build_squeezenet_v11
from common.data_loader import load_data_sample

# =============================================================================
# HARDWARE PARAMETERS - PROVEN WORKING CONFIG (Version 1)
# =============================================================================
GRID_SIZE = 224
GAIN = 128  # Fixed-point scaling factor

# Kernel definitions - FPGA-optimized hand-coded kernels
# These are proven to produce "great looking pictures"

# Ch1: Sharp (3x3) - precise cluster locations
KERNEL_SHARP = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.int16) * GAIN

# Ch2: Medium (3x3) - cluster spread/variance  
KERNEL_MEDIUM = np.array([[1, 2, 1],
                          [2, 8, 2],
                          [1, 2, 1]], dtype=np.int16) * GAIN

# Ch3: Blur (11x11 uniform) - FPGA-optimal: no coefficient LUT needed
# Just count overlap and multiply by GAIN
KERNEL_BLUR = np.ones((11, 11), dtype=np.int16) * GAIN

# Proven working shift values from Version 1
DEFAULT_SHIFTS = (2, 4, 5)  # (ch1, ch2, ch3)

# =============================================================================
# LABEL MAPPING
# =============================================================================
HDF5_TO_MODEL_MAP = {
    1: 2, 3: 5, 4: 7, 5: 4, 12: 0, 13: 1, 14: 3, 23: 6
}
MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]

# =============================================================================
# CORE HARDWARE FUNCTIONS
# =============================================================================

def hardware_gen_layer(iq_samples, kernel, shift_val):
    """
    The Digital Twin of the FPGA Pipeline.
    Strictly integer logic. No floats allowed after coordinate mapping.
    NO DYNAMIC MAX - uses fixed bit-shift for FPGA streaming capability.
    """
    # 1. COORDINATE MAPPING (Float -> Int)
    scale = GRID_SIZE / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale
    
    u_idx = np.clip(np.round(u), 0, GRID_SIZE-1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, GRID_SIZE-1).astype(np.int16)
    
    # 2. ACCUMULATOR (int32 to handle sums without overflow)
    accumulator = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    
    # 3. STAMPING (The Scatter)
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    for x, y in zip(u_idx, v_idx):
        x_min, x_max = max(0, x - pad_h), min(GRID_SIZE, x + pad_h + 1)
        y_min, y_max = max(0, y - pad_w), min(GRID_SIZE, y + pad_w + 1)
        
        k_x_min = pad_h - (x - x_min)
        k_x_max = k_x_min + (x_max - x_min)
        k_y_min = pad_w - (y - y_min)
        k_y_max = k_y_min + (y_max - y_min)
        
        accumulator[x_min:x_max, y_min:y_max] += kernel[k_x_min:k_x_max, k_y_min:k_y_max]
    
    # 4. FUNNEL (Fixed Shift & Clip) - NO DYNAMIC MAX
    output = accumulator >> shift_val
    return np.clip(output, 0, 255).astype(np.uint8)


def generate_hardware_image(iq_samples, shifts=DEFAULT_SHIFTS, mode='multi'):
    """
    Generate image using hardware pipeline.
    
    Args:
        iq_samples: (1024, 2) array of IQ samples
        shifts: tuple of (ch1_shift, ch2_shift, ch3_shift)
        mode: 'multi' for 3-channel, 'single' for 1-channel (Ch1 only)
    
    Returns:
        (224, 224, 3) or (224, 224) array
    """
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, shifts[0])
    
    if mode == 'single':
        return ch1
    
    ch2 = hardware_gen_layer(iq_samples, KERNEL_MEDIUM, shifts[1])
    ch3 = hardware_gen_layer(iq_samples, KERNEL_BLUR, shifts[2])
    return np.stack([ch1, ch2, ch3], axis=-1)


def batch_generate(iq_batch, shifts=DEFAULT_SHIFTS, mode='multi', verbose=False):
    """Batch process IQ samples to images."""
    batch_out = []
    for i, iq in enumerate(iq_batch):
        img = generate_hardware_image(iq, shifts, mode)
        batch_out.append(img)
        
        if verbose and i == 0:
            if mode == 'multi':
                print(f"Sample 0 stats - Ch1: mean={img[:,:,0].mean():.1f}, Ch2: mean={img[:,:,1].mean():.1f}, Ch3: mean={img[:,:,2].mean():.1f}")
            else:
                print(f"Sample 0 stats - mean={img.mean():.1f}, max={img.max()}")
    
    return np.array(batch_out)


def load_filtered_data(data_path, n_samples=100, min_snr=10):
    """Load and filter data to valid classes and SNR range."""
    X, Y, Z = load_data_sample(data_path)
    
    y_int = np.argmax(Y, axis=1) if Y.ndim > 1 else Y.flatten()
    mask = np.isin(y_int, VALID_HDF5_CLASSES) & (Z[:, 0] > min_snr)
    valid_idx = np.where(mask)[0]
    np.random.shuffle(valid_idx)
    
    idx = valid_idx[:n_samples]
    return X[idx], y_int[idx], Z[idx, 0]


# =============================================================================
# MAIN TEST ROUTINE
# =============================================================================

def main():
    print("=== Hardware Image Generation Test ===")
    print(f"Config: GAIN={GAIN}, Kernels: 3x3, 3x3, 11x11, Shifts={DEFAULT_SHIFTS}")
    
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    model_path = 'models/squeezenet_v11_rmsprop.h5'
    
    # Load data
    try:
        print(f"Loading data from {data_path}...")
        samples_iq, samples_y, samples_snr = load_filtered_data(data_path, n_samples=100, min_snr=10)
        print(f"Loaded {len(samples_iq)} samples (SNR > 10)")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Generate images
    print("\nGenerating hardware images...")
    hw_images = batch_generate(samples_iq, shifts=DEFAULT_SHIFTS, mode='multi', verbose=True)
    
    # Save samples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'results_local/hardware_test/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(5, len(hw_images))):
        plt.imsave(f'{save_dir}/sample_{i}.png', hw_images[i])
    print(f"Saved samples to {save_dir}/")
    
    # Test with model if available
    if os.path.exists(model_path):
        print(f"\nLoading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        preds = model.predict(hw_images, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.array([HDF5_TO_MODEL_MAP[y] for y in samples_y])
        
        acc = np.mean(y_true == y_pred)
        print(f"\nAccuracy: {acc*100:.1f}%")
        
        print("\nSample predictions:")
        for i in range(10):
            match = "✓" if y_true[i] == y_pred[i] else "✗"
            print(f"  {i}: True={MODEL_CLASS_NAMES[y_true[i]]:<6} Pred={MODEL_CLASS_NAMES[y_pred[i]]:<6} {match}")
    else:
        print(f"\nModel not found at {model_path}")


if __name__ == "__main__":
    main()
