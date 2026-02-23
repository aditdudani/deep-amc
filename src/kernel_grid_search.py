"""
Kernel Grid Search - Deterministic Kernel Topology Selection for FPGA AMC

Implements the "Funnel Strategy" from the research plan:
1. Test 19 kernel configurations individually (single-channel)
   - Original 15: Point, Cross, Box, Gaussian at various sizes
   - 4 Binomial: Power-of-2 Gaussian approximation (0 DSPs)
2. Evaluate using surrogate training (30% data, 20 epochs, ReduceLROnPlateau)
3. Per-SNR accuracy breakdown to validate inverse correlation hypothesis
4. Output Pareto analysis (Accuracy vs HW Cost)

Key features:
- Deterministic bit-shift calculation (no manual tuning)
- Rolling disk generation (generate → train → delete → next)
- Stratified sampling across (class, SNR) combinations
- CSV output for thesis/paper figures

Usage:
    python kernel_grid_search.py              # Run full search
    python kernel_grid_search.py --resume K05 # Resume from specific kernel
    python kernel_grid_search.py --kernel K03 # Test single kernel
"""

import os
import sys
import math
import json
import shutil
import argparse
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
import h5py
from tqdm import tqdm


# =============================================================================
# CLEAN LOGGING - Skip progress bar updates in log file
# =============================================================================
class TeeLogger:
    """Write to both stdout (with progress bars) and a clean log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        # Skip pure carriage return lines (progress bar updates)
        if '\r' in message and '\n' not in message:
            return  # Don't log progress bar updates
        # Skip lines that look like progress bars
        if any(pattern in message for pattern in ['[====', 'ETA:', '━', 'it/s]']):
            return
        # Write everything else to log
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


# GPU setup
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

print(f"GPUs available: {len(gpus)}")

# =============================================================================
# CONSTANTS
# =============================================================================
GRID_SIZE = 224
GAIN = 128  # Fixed-point scaling for integer arithmetic
MAX_SAMPLES_PER_PIXEL = 1024  # Worst case: all samples hit one pixel

# Dataset parameters
VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]
HDF5_TO_MODEL_MAP = {1: 2, 3: 5, 4: 7, 5: 4, 12: 0, 13: 1, 14: 3, 23: 6}
HDF5_CLASS_NAMES = {
    1: '4ASK', 3: 'BPSK', 4: 'QPSK', 5: '8PSK',
    12: '16QAM', 13: '32QAM', 14: '64QAM', 23: 'OQPSK'
}
MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
NUM_CLASSES = 8
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# Surrogate training config
DATA_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
SUBSET_FRACTION = 0.30  # 30% for statistical validity
TRAIN_VAL_SPLIT = 0.90  # 90/10 train/val
SURROGATE_EPOCHS = 20   # Allows ReduceLROnPlateau (patience=5) to trigger ~2x
BATCH_SIZE = 64
LEARNING_RATE = 0.01
RANDOM_SEED = 42

# Output directories
TEMP_DATA_DIR = 'data/kernel_search_temp'
RESULTS_DIR = 'results_local/kernel_search'


# =============================================================================
# KERNEL DEFINITIONS - The Kernel Search Space
# =============================================================================

def create_box_kernel(size):
    """Uniform/Box kernel - all ones. FPGA-optimal (no multipliers)."""
    return np.ones((size, size), dtype=np.int32) * GAIN

def create_cross_kernel(size):
    """Cross/Manhattan kernel - center row + column only. 50% bandwidth savings."""
    kernel = np.zeros((size, size), dtype=np.int32)
    center = size // 2
    kernel[center, :] = GAIN  # Horizontal line
    kernel[:, center] = GAIN  # Vertical line
    kernel[center, center] = GAIN  # Don't double-count center
    return kernel


def create_center_weighted_cross_kernel(size):
    """Center-weighted Cross: edges=GAIN, center=2×GAIN. 0 DSPs, same write count."""
    kernel = np.zeros((size, size), dtype=np.int32)
    center = size // 2
    kernel[center, :] = GAIN      # edges
    kernel[:, center] = GAIN      # edges
    kernel[center, center] = GAIN * 2  # boosted center (bit-shift left by 1 in HW)
    return kernel

def create_gaussian_kernel(size, sigma=None):
    """
    Gaussian kernel - highest fidelity to KDE theory.
    If sigma not provided, use size/4 (standard heuristic).
    Returns integer kernel scaled by GAIN.
    """
    if sigma is None:
        sigma = size / 4.0
    
    center = size // 2
    y, x = np.ogrid[-center:size-center, -center:size-center]
    kernel_float = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    # Scale so center = GAIN, then convert to integer
    kernel_float = kernel_float / kernel_float.max() * GAIN
    return kernel_float.astype(np.int32)


def create_binomial_kernel(size):
    """
    Hardware-perfect Gaussian approximation using ONLY powers of 2.
    Forces strict power-of-2 decay from center to guarantee 0 DSP usage.
    
    1D pattern: [1, 2, 4, ..., peak, ..., 4, 2, 1] - doubles toward center
    
    Examples:
      3x3: [1, 2, 1] -> peak=2
      5x5: [1, 2, 4, 2, 1] -> peak=4
      7x7: [1, 2, 4, 8, 4, 2, 1] -> peak=8
      11x11: [1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1] -> peak=32
    
    Hardware brilliance: Gaussian-like bell curve at Box kernel cost (0 DSPs).
    """
    # 1. Create 1D array that doubles toward center
    half_size = size // 2
    coeffs = [2**i for i in range(half_size)]
    coeffs_1d = np.array(coeffs + [2**half_size] + coeffs[::-1], dtype=np.int32)
    
    # 2. Create 2D kernel via outer product
    kernel_2d = np.outer(coeffs_1d, coeffs_1d)
    
    # 3. Scale to GAIN (128) using integer operations only
    center_val = kernel_2d[half_size, half_size]
    
    if center_val <= GAIN:
        # Small kernels: multiply up (e.g., 5x5 center=16, scale by 8)
        scale = GAIN // center_val
        kernel_scaled = kernel_2d * scale
    else:
        # Large kernels: shift down (e.g., 11x11 center=1024, shift right 3)
        shift_down = int(np.log2(center_val // GAIN))
        kernel_scaled = kernel_2d >> shift_down
    
    return kernel_scaled.astype(np.int32)
    

def define_kernel_configs():
    """
    Define all kernel configurations for the grid search.
    Base: 15 (Point/Cross/Box/Gaussian)
    Binomial: 4 (power-of-2 Gaussian approximations)
    Center-Weighted Cross: 4 (boosted center, same bandwidth writes)
    Returns dict: config_id -> {kernel, size, topology, cost_proxy, uses_dsp}
    """
    configs = {}
    
    # K01: 1x1 Point (baseline - minimal logic)
    configs['K01'] = {
        'name': '1x1_Point',
        'kernel': np.array([[GAIN]], dtype=np.int32),
        'size': 1,
        'topology': 'Point',
        'cost_proxy': 1,
        'uses_dsp': False,
    }
    
    # K02: 3x3 Cross (minimal bandwidth)
    configs['K02'] = {
        'name': '3x3_Cross',
        'kernel': create_cross_kernel(3),
        'size': 3,
        'topology': 'Cross',
        'cost_proxy': 5,
        'uses_dsp': False,
    }
    
    # K03: 3x3 Box (uniform local spread)
    configs['K03'] = {
        'name': '3x3_Box',
        'kernel': create_box_kernel(3),
        'size': 3,
        'topology': 'Box',
        'cost_proxy': 9,
        'uses_dsp': False,
    }
    
    # K04: 3x3 Gaussian (high-fidelity, tests if smoothness matters)
    configs['K04'] = {
        'name': '3x3_Gaussian',
        'kernel': create_gaussian_kernel(3),
        'size': 3,
        'topology': 'Gaussian',
        'cost_proxy': 18,  # 9 × 2 (DSP penalty)
        'uses_dsp': True,
    }
    
    # K05: 5x5 Cross (mid-range bandwidth saver)
    configs['K05'] = {
        'name': '5x5_Cross',
        'kernel': create_cross_kernel(5),
        'size': 5,
        'topology': 'Cross',
        'cost_proxy': 9,
        'uses_dsp': False,
    }
    
    # K06: 5x5 Box (mid-range uniform)
    configs['K06'] = {
        'name': '5x5_Box',
        'kernel': create_box_kernel(5),
        'size': 5,
        'topology': 'Box',
        'cost_proxy': 25,
        'uses_dsp': False,
    }
    
    # K07: 5x5 Gaussian (mid-range fidelity)
    configs['K07'] = {
        'name': '5x5_Gaussian',
        'kernel': create_gaussian_kernel(5),
        'size': 5,
        'topology': 'Gaussian',
        'cost_proxy': 50,  # 25 × 2 (DSP penalty)
        'uses_dsp': True,
    }
    
    # K08: 7x7 Cross (large bandwidth saver)
    configs['K08'] = {
        'name': '7x7_Cross',
        'kernel': create_cross_kernel(7),
        'size': 7,
        'topology': 'Cross',
        'cost_proxy': 13,
        'uses_dsp': False,
    }
    
    # K09: 7x7 Box (large uniform)
    configs['K09'] = {
        'name': '7x7_Box',
        'kernel': create_box_kernel(7),
        'size': 7,
        'topology': 'Box',
        'cost_proxy': 49,
        'uses_dsp': False,
    }
    
    # K10: 7x7 Gaussian (large fidelity)
    configs['K10'] = {
        'name': '7x7_Gaussian',
        'kernel': create_gaussian_kernel(7),
        'size': 7,
        'topology': 'Gaussian',
        'cost_proxy': 98,  # 49 × 2 (DSP penalty)
        'uses_dsp': True,
    }
    
    # K11: 11x11 Cross (very large bandwidth saver)
    configs['K11'] = {
        'name': '11x11_Cross',
        'kernel': create_cross_kernel(11),
        'size': 11,
        'topology': 'Cross',
        'cost_proxy': 21,
        'uses_dsp': False,
    }
    
    # K12: 11x11 Box (current baseline, strong spatial integrator)
    configs['K12'] = {
        'name': '11x11_Box',
        'kernel': create_box_kernel(11),
        'size': 11,
        'topology': 'Box',
        'cost_proxy': 121,
        'uses_dsp': False,
    }
    
    # K13: 11x11 Gaussian (max fidelity approx of software "fog")
    configs['K13'] = {
        'name': '11x11_Gaussian',
        'kernel': create_gaussian_kernel(11),
        'size': 11,
        'topology': 'Gaussian',
        'cost_proxy': 242,  # 121 × 2 (DSP penalty)
        'uses_dsp': True,
    }
    
    # K14: 15x15 Box (extreme uniform spread, tests integration limit)
    configs['K14'] = {
        'name': '15x15_Box',
        'kernel': create_box_kernel(15),
        'size': 15,
        'topology': 'Box',
        'cost_proxy': 225,
        'uses_dsp': False,
    }
    
    # K15: 15x15 Gaussian (extreme fidelity upper bound)
    configs['K15'] = {
        'name': '15x15_Gaussian',
        'kernel': create_gaussian_kernel(15),
        'size': 15,
        'topology': 'Gaussian',
        'cost_proxy': 450,  # 225 × 2 (DSP penalty)
        'uses_dsp': True,
    }
    
    # =========================================================================
    # BINOMIAL KERNELS - Power-of-2 Gaussian approximation (0 DSPs)
    # =========================================================================
    
    # K16: 3x3 Binomial (exact: [[1,2,1],[2,4,2],[1,2,1]] × GAIN)
    configs['K16'] = {
        'name': '3x3_Binomial',
        'kernel': create_binomial_kernel(3),
        'size': 3,
        'topology': 'Binomial',
        'cost_proxy': 9,  # 9 writes, 0 DSPs (all bit-shifts)
        'uses_dsp': False,
    }
    
    # K17: 5x5 Binomial (power-of-2 approximation)
    configs['K17'] = {
        'name': '5x5_Binomial',
        'kernel': create_binomial_kernel(5),
        'size': 5,
        'topology': 'Binomial',
        'cost_proxy': 25,  # 25 writes, 0 DSPs
        'uses_dsp': False,
    }
    
    # K18: 7x7 Binomial (power-of-2 approximation)
    configs['K18'] = {
        'name': '7x7_Binomial',
        'kernel': create_binomial_kernel(7),
        'size': 7,
        'topology': 'Binomial',
        'cost_proxy': 49,  # 49 writes, 0 DSPs
        'uses_dsp': False,
    }
    
    # K19: 11x11 Binomial (power-of-2 approximation)
    configs['K19'] = {
        'name': '11x11_Binomial',
        'kernel': create_binomial_kernel(11),
        'size': 11,
        'topology': 'Binomial',
        'cost_proxy': 121,  # 121 writes, 0 DSPs
        'uses_dsp': False,
    }

    # Center-Weighted Cross (edges=GAIN, center=2×GAIN) - same write count as Cross
    configs['K20'] = {
        'name': '3x3_Cross_Centered',
        'kernel': create_center_weighted_cross_kernel(3),
        'size': 3,
        'topology': 'CrossCentered',
        'cost_proxy': 5,
        'uses_dsp': False,
    }

    configs['K21'] = {
        'name': '5x5_Cross_Centered',
        'kernel': create_center_weighted_cross_kernel(5),
        'size': 5,
        'topology': 'CrossCentered',
        'cost_proxy': 9,
        'uses_dsp': False,
    }

    configs['K22'] = {
        'name': '7x7_Cross_Centered',
        'kernel': create_center_weighted_cross_kernel(7),
        'size': 7,
        'topology': 'CrossCentered',
        'cost_proxy': 13,
        'uses_dsp': False,
    }

    configs['K23'] = {
        'name': '11x11_Cross_Centered',
        'kernel': create_center_weighted_cross_kernel(11),
        'size': 11,
        'topology': 'CrossCentered',
        'cost_proxy': 21,
        'uses_dsp': False,
    }
    
    return configs


def calculate_shift(kernel):
    """
    Empirically calibrated bit-shift calculation for a kernel.
    
    CRITICAL FIX: The document's formula assumed worst-case (1024 samples in 1 pixel),
    but real signals spread samples across 200+ pixels with max ~20-50 overlap.
    
    Empirically calibrated approach:
    - Small kernels (1-5): shift = 0 (accumulator rarely exceeds 255)
    - Medium kernels (7): shift = 1-2
    - Large kernels (11-15): shift = 3-5 (significant overlap/spread)
    
    The working phase2_channel_comparison used shifts of (0, 0, 3) for 3x3, 3x3, 11x11.
    """
    kernel_sum = int(kernel.sum())
    k_size = kernel.shape[0]
    
    # Empirical calibration based on kernel size
    # Larger kernels cause more overlap, need more shifting
    if k_size <= 3:
        # 1x1 or 3x3: minimal overlap, shift 0
        return 0
    elif k_size <= 5:
        # 5x5: slight overlap, shift 1
        return 1
    elif k_size <= 7:
        # 7x7: moderate overlap, shift 2
        return 2
    elif k_size <= 11:
        # 11x11: significant overlap, shift 3
        return 3
    else:
        # 15x15+: extensive overlap, shift 4
        return 4


# =============================================================================
# HARDWARE IMAGE GENERATION
# =============================================================================

def hw_gen_layer(iq_samples, kernel, shift_val):
    """
    Hardware-accurate image generation with integer-only arithmetic.
    Identical to simulate_hardware.py but parameterized for any kernel.
    """
    scale = GRID_SIZE / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale
    
    u_idx = np.clip(np.round(u), 0, GRID_SIZE-1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, GRID_SIZE-1).astype(np.int16)
    
    # int32 accumulator to prevent overflow
    accumulator = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    
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
    
    # Simulate FPGA int16 clamping (per Risk Section 9.3 in Definitive Plan)
    # This ensures model trains on the exact artifacts that might occur in hardware
    accumulator = np.clip(accumulator, -32768, 32767)
    
    # Apply bit-shift and clip to 8-bit
    output = accumulator >> shift_val
    return np.clip(output, 0, 255).astype(np.uint8)


def generate_single_channel_rgb(iq_samples, kernel, shift_val):
    """Generate single-channel image stacked to RGB for model compatibility."""
    ch = hw_gen_layer(iq_samples, kernel, shift_val)
    return np.stack([ch, ch, ch], axis=-1)


# =============================================================================
# SQUEEZENET MODEL (exact copy from proper_hw_train_test.py)
# =============================================================================

def _fire_module(x, squeeze_channels, expand_channels):
    squeeze = layers.Conv2D(squeeze_channels, (1, 1), activation='relu', padding='valid',
                            kernel_initializer='he_normal')(x)
    expand_1x1 = layers.Conv2D(expand_channels, (1, 1), activation='relu', padding='valid',
                               kernel_initializer='he_normal')(squeeze)
    expand_3x3 = layers.Conv2D(expand_channels, (3, 3), activation='relu', padding='same',
                               kernel_initializer='he_normal')(squeeze)
    return layers.Concatenate(axis=-1)([expand_1x1, expand_3x3])


def build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=8, dropout_rate=0.0):
    """SqueezeNet v1.1 - proven working architecture."""
    inputs = layers.Input(shape=input_shape, dtype='float32')
    x = layers.Rescaling(1.0 / 255.0, name='rescale_0_1')(inputs)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu',
                      name='conv1', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)
    x = _fire_module(x, squeeze_channels=16, expand_channels=64)
    x = _fire_module(x, squeeze_channels=16, expand_channels=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool3')(x)
    x = _fire_module(x, squeeze_channels=32, expand_channels=128)
    x = _fire_module(x, squeeze_channels=32, expand_channels=128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool5')(x)
    x = _fire_module(x, squeeze_channels=48, expand_channels=192)
    x = _fire_module(x, squeeze_channels=48, expand_channels=192)
    x = _fire_module(x, squeeze_channels=64, expand_channels=256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool8')(x)
    x = _fire_module(x, squeeze_channels=64, expand_channels=256)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='dropout')(x)
    # CRITICAL: activation=None here, softmax is separate
    x = layers.Conv2D(num_classes, (1, 1), activation=None, padding='valid',
                      name='conv_final', dtype='float32', kernel_initializer='he_normal')(x)
    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    outputs = layers.Softmax(name='predictions')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='squeezenet_v1_1')


# =============================================================================
# DATASET GENERATION (Rolling Disk)
# =============================================================================

def generate_dataset_to_disk(kernel, shift_val, output_dir, seed=RANDOM_SEED):
    """
    Generate images for a single kernel configuration to disk.
    Uses stratified sampling across (class, SNR) combinations.
    """
    print(f"  Generating images to {output_dir}...")
    
    # Clear output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create directories for each class
    for split in ['train', 'validation']:
        for class_name in MODEL_CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    with h5py.File(DATA_PATH, 'r') as hf:
        X = hf['X']
        Y_onehot = hf['Y'][:]
        y_int = np.argmax(Y_onehot, axis=1)
        Z_2d = hf['Z'][:]
        snrs = (Z_2d[:, 0] if Z_2d.ndim > 1 else Z_2d.flatten()).astype(np.int32)
        
        np.random.seed(seed)
        
        # Stratified sampling by (class, SNR)
        samples_per_combo = int(4096 * SUBSET_FRACTION)
        all_selected = []
        
        for cls in VALID_HDF5_CLASSES:
            for snr in TARGET_SNRS:
                mask = (y_int == cls) & (snrs == snr)
                indices = np.where(mask)[0]
                n_take = min(samples_per_combo, len(indices))
                selected = np.random.choice(indices, n_take, replace=False)
                all_selected.extend([(idx, cls, snr) for idx in selected])
        
        np.random.shuffle(all_selected)
        
        # Group by class for train/val splitting
        by_class = {cls: [] for cls in VALID_HDF5_CLASSES}
        for idx, cls, snr in all_selected:
            by_class[cls].append((idx, snr))
        
        # Generate images
        image_counts = {'train': 0, 'validation': 0}
        snr_counts = {snr: {'train': 0, 'val': 0} for snr in TARGET_SNRS}
        
        for hdf5_cls, items in by_class.items():
            class_name = HDF5_CLASS_NAMES[hdf5_cls]
            np.random.seed(seed + hdf5_cls)
            np.random.shuffle(items)
            
            split_point = int(len(items) * TRAIN_VAL_SPLIT)
            train_items = items[:split_point]
            val_items = items[split_point:]
            
            # Training images
            for idx, snr in tqdm(train_items, desc=f"  train/{class_name}", leave=False):
                iq = np.asarray(X[idx], dtype=np.float32)
                img = generate_single_channel_rgb(iq, kernel, shift_val)
                fname = f"snr{snr:+03d}_idx{idx:07d}.png"
                tf.keras.utils.save_img(
                    os.path.join(output_dir, 'train', class_name, fname), img
                )
                image_counts['train'] += 1
                snr_counts[snr]['train'] += 1
            
            # Validation images
            for idx, snr in tqdm(val_items, desc=f"  val/{class_name}", leave=False):
                iq = np.asarray(X[idx], dtype=np.float32)
                img = generate_single_channel_rgb(iq, kernel, shift_val)
                fname = f"snr{snr:+03d}_idx{idx:07d}.png"
                tf.keras.utils.save_img(
                    os.path.join(output_dir, 'validation', class_name, fname), img
                )
                image_counts['validation'] += 1
                snr_counts[snr]['val'] += 1
    
    print(f"  Generated {image_counts['train']} train, {image_counts['validation']} val images")
    return image_counts, snr_counts


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def make_dataset(data_dir, batch_size, shuffle=True):
    """Create tf.data dataset for streaming from disk."""
    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False
    
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    class_names = ds.class_names
    ds = ds.prefetch(AUTOTUNE).with_options(options)
    return ds, class_names


def train_and_evaluate(config_id, config, output_dir):
    """
    Train SqueezeNet on generated images and evaluate per-SNR accuracy.
    Returns dict with overall and per-SNR metrics.
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    
    print(f"  Training SqueezeNet for {SURROGATE_EPOCHS} epochs...")
    
    # Create datasets
    train_ds, class_names = make_dataset(train_dir, BATCH_SIZE, shuffle=True)
    val_ds, _ = make_dataset(val_dir, BATCH_SIZE, shuffle=False)
    
    # Build model
    tf.keras.backend.clear_session()
    model = build_squeezenet_v11(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES,
        dropout_rate=0.0
    )
    
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    # Callbacks - ReduceLROnPlateau for stable convergence (per Section 5.4)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train with LR scheduling
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=SURROGATE_EPOCHS,
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Get overall metrics
    overall_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"  Overall val_acc: {overall_val_acc*100:.2f}% (best: {best_val_acc*100:.2f}%)")
    
    # Per-SNR evaluation
    print(f"  Evaluating per-SNR accuracy...")
    snr_accuracies = evaluate_per_snr(model, val_dir, class_names)
    
    results = {
        'config_id': config_id,
        'name': config['name'],
        'size': config['size'],
        'topology': config['topology'],
        'cost_proxy': config['cost_proxy'],
        'uses_dsp': config['uses_dsp'],
        'overall_val_acc': float(overall_val_acc),
        'best_val_acc': float(best_val_acc),
        'final_val_loss': float(final_val_loss),
        'per_snr_accuracy': snr_accuracies,
    }
    
    # Print per-SNR results
    print(f"  Per-SNR accuracy:")
    for snr in TARGET_SNRS:
        acc = snr_accuracies.get(str(snr), 0)
        print(f"    SNR {snr:>3}dB: {acc*100:.1f}%")
    
    return results


def evaluate_per_snr(model, val_dir, class_names):
    """
    Evaluate model accuracy for each SNR level.
    Uses filename pattern to extract SNR: snr+XX_idx*.png
    Batched for efficiency.
    """
    snr_correct = {snr: 0 for snr in TARGET_SNRS}
    snr_total = {snr: 0 for snr in TARGET_SNRS}
    
    # Build class name to index mapping
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Collect all images with their metadata
    all_images = []  # (img_path, true_label, snr)
    
    for class_name in MODEL_CLASS_NAMES:
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        true_label = class_to_idx[class_name]
        
        for fname in os.listdir(class_dir):
            if not fname.endswith('.png'):
                continue
            
            # Extract SNR from filename: snr+XX_idx*.png
            try:
                snr_str = fname.split('_')[0]  # snr+XX
                snr = int(snr_str.replace('snr', ''))
            except:
                continue
            
            if snr not in TARGET_SNRS:
                continue
            
            img_path = os.path.join(class_dir, fname)
            all_images.append((img_path, true_label, snr))
    
    # Process in batches for efficiency
    batch_size = 64
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i+batch_size]
        
        # Load batch
        img_arrays = []
        for img_path, _, _ in batch:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_arrays.append(img_array)
        
        img_batch = np.stack(img_arrays, axis=0)
        
        # Batch predict
        preds = model.predict(img_batch, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        
        # Accumulate results
        for j, (_, true_label, snr) in enumerate(batch):
            snr_total[snr] += 1
            if pred_labels[j] == true_label:
                snr_correct[snr] += 1
    
    # Calculate accuracies
    snr_accuracies = {}
    for snr in TARGET_SNRS:
        if snr_total[snr] > 0:
            snr_accuracies[str(snr)] = snr_correct[snr] / snr_total[snr]
        else:
            snr_accuracies[str(snr)] = 0.0
    
    return snr_accuracies


# =============================================================================
# RESULTS AND ANALYSIS
# =============================================================================

def save_results(all_results, output_path):
    """Save results to JSON and CSV formats."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save JSON (full details)
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed results to {json_path}")
    
    # Save CSV (for easy analysis)
    csv_lines = ['config_id,name,size,topology,cost_proxy,uses_dsp,overall_acc,best_acc,' + 
                 ','.join([f'snr_{s}dB' for s in TARGET_SNRS])]
    
    for r in all_results:
        snr_accs = [f"{r['per_snr_accuracy'].get(str(s), 0)*100:.2f}" for s in TARGET_SNRS]
        line = f"{r['config_id']},{r['name']},{r['size']},{r['topology']},{r['cost_proxy']}," + \
               f"{r['uses_dsp']},{r['overall_val_acc']*100:.2f},{r['best_val_acc']*100:.2f}," + \
               ','.join(snr_accs)
        csv_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"Saved rankings to {output_path}")


def print_pareto_analysis(all_results):
    """Print Pareto frontier analysis."""
    print("\n" + "="*70)
    print("PARETO ANALYSIS - Accuracy vs HW Cost")
    print("="*70)
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True)
    
    print("\nRanked by Overall Accuracy:")
    print(f"{'Rank':<5} {'Config':<15} {'Accuracy':<10} {'Cost':<8} {'DSP?':<6}")
    print("-" * 50)
    
    for i, r in enumerate(sorted_results, 1):
        print(f"{i:<5} {r['name']:<15} {r['best_val_acc']*100:>7.2f}%  {r['cost_proxy']:<8} {str(r['uses_dsp']):<6}")
    
    # Find Pareto-optimal configurations
    print("\nPareto-Optimal Configurations (non-dominated):")
    pareto = []
    for r in sorted_results:
        dominated = False
        for other in sorted_results:
            if other['best_val_acc'] > r['best_val_acc'] and other['cost_proxy'] <= r['cost_proxy']:
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    
    for r in pareto:
        print(f"  {r['name']}: {r['best_val_acc']*100:.2f}% @ cost={r['cost_proxy']}")
    
    # Per-SNR champions (validates inverse correlation hypothesis)
    print("\nPer-SNR Champions (validates kernel-SNR inverse correlation):")
    for snr in TARGET_SNRS:
        best = max(all_results, key=lambda x: x['per_snr_accuracy'].get(str(snr), 0))
        acc = best['per_snr_accuracy'].get(str(snr), 0)
        print(f"  SNR {snr:>3}dB: {best['name']} ({acc*100:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

def load_existing_results(csv_path):
    """Load existing results from CSV file."""
    results = []
    if not os.path.exists(csv_path):
        return results
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return results
    
    # Parse header to get column indices
    header = lines[0].strip().split(',')
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue
        
        result = {
            'config_id': parts[0],
            'name': parts[1],
            'size': int(parts[2]),
            'topology': parts[3],
            'cost_proxy': int(parts[4]),
            'uses_dsp': parts[5] == 'True',
            'overall_val_acc': float(parts[6]) / 100,
            'best_val_acc': float(parts[7]) / 100,
            'per_snr_accuracy': {}
        }
        
        # Parse per-SNR accuracies
        for i, snr in enumerate(TARGET_SNRS):
            if 8 + i < len(parts):
                result['per_snr_accuracy'][str(snr)] = float(parts[8 + i]) / 100
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Kernel Grid Search for FPGA AMC')
    parser.add_argument('--resume', type=str, help='Resume from specific kernel (e.g., K05)')
    parser.add_argument('--kernel', type=str, help='Test single kernel only (e.g., K03)')
    parser.add_argument(
        '--kernels',
        nargs='+',
        help='Test multiple kernels (supports "K16,K17" or "K16 K17" or mixed comma+space)'
    )
    parser.add_argument('--append', type=str, help='Append results to existing CSV file')
    parser.add_argument('--no-log', action='store_true', help='Disable clean log file')
    args = parser.parse_args()
    
    # Set up clean logging (filters progress bars)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = None
    if not args.no_log:
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/kernel_grid_search_{timestamp}.log'
        logger = TeeLogger(log_path)
        sys.stdout = logger
        print(f"Clean log: {log_path}")
    
    print("\n" + "="*70)
    print("KERNEL GRID SEARCH - Deterministic Kernel Topology Selection")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {DATA_PATH}")
    print(f"Subset: {SUBSET_FRACTION*100:.0f}%")
    print(f"Epochs: {SURROGATE_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Optimizer: SGD(lr={LEARNING_RATE}, momentum=0.9)")
    print("="*70)
    
    # Check data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    
    # Define kernel configurations
    configs = define_kernel_configs()
    
    # Print kernel info
    print(f"\n{len(configs)} kernel configurations to test:")
    for config_id, config in configs.items():
        shift = calculate_shift(config['kernel'])
        ks = config['kernel'].sum()
        print(f"  {config_id}: {config['name']:<15} | sum={ks:>8} | shift={shift} | cost={config['cost_proxy']}")
    
    # Filter kernels if requested
    if args.kernel:
        if args.kernel not in configs:
            print(f"ERROR: Unknown kernel {args.kernel}")
            return
        configs = {args.kernel: configs[args.kernel]}
    elif args.kernels:
        kernel_list = []
        for token in args.kernels:
            kernel_list.extend([k.strip() for k in token.split(',') if k.strip()])
        invalid = [k for k in kernel_list if k not in configs]
        if invalid:
            print(f"ERROR: Unknown kernels: {invalid}")
            return
        configs = {k: configs[k] for k in kernel_list}
        print(f"\nTesting {len(configs)} selected kernels: {kernel_list}")
    
    # Resume handling
    start_idx = 0
    if args.resume:
        kernel_ids = list(configs.keys())
        if args.resume in kernel_ids:
            start_idx = kernel_ids.index(args.resume)
            print(f"\nResuming from {args.resume} (index {start_idx})")
    
    # Load existing results if appending
    all_results = []
    if args.append:
        if os.path.exists(args.append):
            all_results = load_existing_results(args.append)
            existing_ids = {r['config_id'] for r in all_results}
            print(f"\nLoaded {len(all_results)} existing results from {args.append}")
            print(f"  Existing kernels: {sorted(existing_ids)}")
        else:
            print(f"\nAppend file not found, will create: {args.append}")
    
    kernel_ids = list(configs.keys())[start_idx:]
    
    for i, config_id in enumerate(kernel_ids):
        config = configs[config_id]
        shift_val = calculate_shift(config['kernel'])
        
        print(f"\n{'='*70}")
        print(f"[{i+1+start_idx}/{len(configs)}] Testing {config_id}: {config['name']}")
        print(f"  Size: {config['size']}x{config['size']}, Topology: {config['topology']}")
        print(f"  Kernel sum: {config['kernel'].sum()}, Calculated shift: {shift_val}")
        print(f"  HW Cost proxy: {config['cost_proxy']}, Uses DSP: {config['uses_dsp']}")
        print("="*70)
        
        # Step 1: Generate images
        generate_dataset_to_disk(
            config['kernel'],
            shift_val,
            TEMP_DATA_DIR,
            seed=RANDOM_SEED
        )
        
        # Step 2: Train and evaluate
        results = train_and_evaluate(config_id, config, TEMP_DATA_DIR)
        results['calculated_shift'] = shift_val
        all_results.append(results)
        
        # Step 3: Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_path = args.append if args.append else os.path.join(RESULTS_DIR, 'kernel_rankings.csv')
        save_results(all_results, intermediate_path)
        
        # Step 4: Clean up temp directory
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)
            print(f"  Cleaned up temp directory")
    
    # Final analysis
    print_pareto_analysis(all_results)
    
    # Save final results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.append:
        # Also save a timestamped copy
        base_path = args.append.replace('.csv', f'_{timestamp}.csv')
        save_results(all_results, base_path)
        # Update the original append file
        save_results(all_results, args.append)
    else:
        final_path = os.path.join(RESULTS_DIR, f'kernel_rankings_{timestamp}.csv')
        save_results(all_results, final_path)
    
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    print(f"Results saved to: {RESULTS_DIR}/")
    print("Next step: Use rankings to select champion kernel for Phase 2")
    
    # Close logger
    if logger:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
