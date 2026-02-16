"""
Test: Try larger Gaussian-like blur kernels to make HW images more discriminative.
The current 3x3 kernels create sparse images that look identical across classes.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from tqdm import tqdm

print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")

GRID_SIZE = 224
VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]
HDF5_CLASS_NAMES = {1: '4ASK', 3: 'BPSK', 4: 'QPSK', 5: '8PSK', 
                    12: '16QAM', 13: '32QAM', 14: '64QAM', 23: 'OQPSK'}
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# =============================================================================
# IMPROVED HARDWARE IMAGE GENERATION - LARGER GAUSSIAN BLUR
# =============================================================================

def create_gaussian_kernel(size, sigma):
    """Create a Gaussian kernel for smoother, more distinctive images."""
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    # Normalize to sum to 1, then scale
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def hardware_gen_gaussian(iq_samples, kernel, normalize=True):
    """Hardware-style generation with Gaussian blur for better discrimination."""
    scale = GRID_SIZE / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale
    
    u_idx = np.clip(np.round(u), 0, GRID_SIZE-1).astype(np.int32)
    v_idx = np.clip(np.round(v), 0, GRID_SIZE-1).astype(np.int32)
    
    accumulator = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
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
    
    if normalize:
        max_val = accumulator.max()
        if max_val > 0:
            accumulator = accumulator / max_val
    
    return accumulator

def generate_image_gaussian(iq_samples, sigma=5.0, kernel_size=31):
    """Generate single-channel image with Gaussian blur."""
    kernel = create_gaussian_kernel(kernel_size, sigma)
    ch = hardware_gen_gaussian(iq_samples, kernel, normalize=True)
    return np.stack([ch, ch, ch], axis=-1)  # RGB for model compatibility


# =============================================================================
# MODEL (same as before)
# =============================================================================
def build_squeezenet_simple(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    def fire_module(x, squeeze, expand):
        squeeze_out = layers.Conv2D(squeeze, (1, 1), activation='relu', padding='same')(x)
        expand_1x1 = layers.Conv2D(expand, (1, 1), activation='relu', padding='same')(squeeze_out)
        expand_3x3 = layers.Conv2D(expand, (3, 3), activation='relu', padding='same')(squeeze_out)
        return layers.Concatenate()([expand_1x1, expand_3x3])
    
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = fire_module(x, 32, 128)
    x = fire_module(x, 32, 128)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = fire_module(x, 64, 256)
    
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax')(x)
    
    return keras.Model(inputs, outputs)


# =============================================================================
# MAIN
# =============================================================================
def main():
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    
    print("\n" + "="*60)
    print("TEST: GAUSSIAN BLUR HARDWARE IMAGES")
    print("="*60)
    
    # Load data
    print(f"\nLoading from {data_path}...")
    with h5py.File(data_path, 'r') as f:
        X_all = f['X'][:]
        Y_all = f['Y'][:]
        Z_all = f['Z'][:].flatten()
    
    y_int = np.argmax(Y_all, axis=1)
    
    # Filter to valid classes and SNRs
    mask = np.isin(y_int, VALID_HDF5_CLASSES) & np.isin(Z_all, TARGET_SNRS)
    X_all = X_all[mask]
    y_int = y_int[mask]
    Z_all = Z_all[mask]
    
    print(f"  Filtered samples: {len(X_all)}")
    
    # Take small subset for quick test
    n_samples = 8000  # 1000 per class
    indices = np.random.permutation(len(X_all))[:n_samples]
    X_subset = X_all[indices]
    y_subset = y_int[indices]
    
    # Map HDF5 class indices to model indices (0-7)
    HDF5_TO_MODEL = {1: 0, 3: 1, 4: 2, 5: 3, 12: 4, 13: 5, 14: 6, 23: 7}
    y_model = np.array([HDF5_TO_MODEL[y] for y in y_subset])
    class_names = ['4ASK', 'BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM', 'OQPSK']
    
    # Generate images with Gaussian blur
    print("\nGenerating images with Gaussian blur (sigma=5, kernel=31x31)...")
    images = []
    for i in tqdm(range(len(X_subset))):
        img = generate_image_gaussian(X_subset[i], sigma=5.0, kernel_size=31)
        images.append(img)
    
    X_images = np.array(images, dtype=np.float32)
    
    # Split train/val
    n_train = int(len(X_images) * 0.8)
    perm = np.random.permutation(len(X_images))
    X_train = X_images[perm[:n_train]]
    y_train = y_model[perm[:n_train]]
    X_val = X_images[perm[n_train:]]
    y_val = y_model[perm[n_train:]]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Check image statistics
    print("\n" + "="*60)
    print("IMAGE STATISTICS (Gaussian blur)")
    print("="*60)
    print(f"  Overall - mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    for i, cls in enumerate(class_names):
        cls_mask = y_train == i
        if cls_mask.sum() > 0:
            cls_mean = X_train[cls_mask].mean()
            cls_std = X_train[cls_mask].std()
            print(f"  {cls:8s} - mean: {cls_mean:.4f}, std: {cls_std:.4f}, count: {cls_mask.sum()}")
    
    # Build and train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    model = build_squeezenet_simple((GRID_SIZE, GRID_SIZE, 3), len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Model params: {model.count_params():,}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Final Val Accuracy: {final_val_acc*100:.2f}%")
    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
    
    if best_val_acc > 0.20:
        print("\n  SUCCESS! Gaussian blur makes images learnable!")
        print("  The issue was the sparse 3x3 kernel - not enough discrimination.")
    else:
        print("\n  STILL STUCK - may need even larger blur or different approach.")


if __name__ == "__main__":
    main()
