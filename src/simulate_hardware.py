"""
Hardware Image Generation - Digital Twin for FPGA AMC Pipeline
Generates constellation images using integer-only local kernel stamping.
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
from common.image_generator import tf_generate_three_channel_image

# --- HARDWARE PARAMETERS ---
GRID_SIZE = 224
GAIN = 32

# Hand-crafted integer kernels (from hardware proposal)
# These have meaningful spread, unlike exp(-alpha*r) with high alpha
KERNEL_CH1 = (np.array([[0, 1, 0],
                        [1, 4, 1],
                        [0, 1, 0]], dtype=np.int16) * GAIN)  # Sharp

KERNEL_CH2 = (np.array([[1, 2, 1],
                        [2, 8, 2],
                        [1, 2, 1]], dtype=np.int16) * GAIN)  # Medium

KERNEL_CH3 = (np.ones((5, 5), dtype=np.int16) * GAIN)  # Blur/Wide

# Label mapping: HDF5 index -> Model index
HDF5_TO_MODEL_MAP = {
    1: 2, 3: 5, 4: 7, 5: 4, 12: 0, 13: 1, 14: 3, 23: 6
}
MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']


def hardware_gen_layer(iq_samples, kernel, shift_val):
    """Hardware-accurate image generation layer with fixed bit-shift scaling."""
    scale = GRID_SIZE / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale
    
    u_idx = np.clip(np.round(u), 0, GRID_SIZE-1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, GRID_SIZE-1).astype(np.int16)
    
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
    
    output = accumulator >> shift_val
    return np.clip(output, 0, 255).astype(np.uint8)


def generate_hardware_image(iq_samples, shifts=(2, 3, 4), mode='multi'):
    """Generate image using hardware pipeline. Mode: 'single' or 'multi'."""
    ch1 = hardware_gen_layer(iq_samples, KERNEL_CH1, shifts[0])
    
    if mode == 'single':
        return ch1
    
    ch2 = hardware_gen_layer(iq_samples, KERNEL_CH2, shifts[1])
    ch3 = hardware_gen_layer(iq_samples, KERNEL_CH3, shifts[2])
    return np.stack([ch1, ch2, ch3], axis=-1)


def batch_generate(iq_batch, shifts=(2, 3, 4), mode='multi', verbose=False):
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


def main():
    print("=== Hardware Image Generation Test ===")
    
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    model_path = 'models/squeezenet_v11_rmsprop.h5'
    
    # Load data
    try:
        print(f"Loading data from {data_path}...")
        X, Y, Z = load_data_sample(data_path)
        
        y_int = np.argmax(Y, axis=1) if Y.ndim > 1 else Y.flatten()
        valid_classes = [1, 3, 4, 5, 12, 13, 14, 23]
        mask = np.isin(y_int, valid_classes) & (Z[:, 0] > 10)
        valid_idx = np.where(mask)[0]
        np.random.shuffle(valid_idx)
        
        samples_iq = X[valid_idx[:100]]
        samples_y = y_int[valid_idx[:100]]
        print(f"Loaded {len(samples_iq)} samples (SNR > 10)")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Generate images
    print("\nGenerating hardware images...")
    shifts = (2, 3, 4)  # Will be tuned by calibration script
    hw_images = batch_generate(samples_iq, shifts=shifts, mode='multi', verbose=True)
    
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


if __name__ == "__main__":
    main()
