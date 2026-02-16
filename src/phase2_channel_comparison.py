"""
Phase 2: Architecture Decision - Single vs Multi-Channel Comparison

This script:
1. Generates 10% subset with SINGLE-channel (Ch1 only, 224x224x1)
2. Generates 10% subset with MULTI-channel (3-channel, 224x224x3)
3. Trains SqueezeNet for 10 epochs on each
4. Compares accuracy to decide optimal architecture

If single-channel is within 5% of multi-channel → go single for FPGA savings.
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from datetime import datetime
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.data_loader import load_data_sample

# =============================================================================
# HARDWARE PARAMETERS (MUST MATCH simulate_hardware.py / calibrate_hardware.py)
# =============================================================================
GRID_SIZE = 224
GAIN = 128

# Kernels
KERNEL_SHARP = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.int16) * GAIN

KERNEL_MEDIUM = np.array([[1, 2, 1],
                          [2, 8, 2],
                          [1, 2, 1]], dtype=np.int16) * GAIN

KERNEL_BLUR = np.ones((11, 11), dtype=np.int16) * GAIN

# Calibrated shifts from Phase 1
DEFAULT_SHIFTS = (0, 0, 3)

# Label mapping
VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]
HDF5_TO_MODEL_MAP = {1: 2, 3: 5, 4: 7, 5: 4, 12: 0, 13: 1, 14: 3, 23: 6}
MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
NUM_CLASSES = 8

# =============================================================================
# HARDWARE IMAGE GENERATION
# =============================================================================

def hardware_gen_layer(iq_samples, kernel, shift_val):
    """Hardware-accurate image generation layer."""
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


def generate_single_channel(iq_samples):
    """Generate single-channel image (Ch1 sharp only)."""
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, DEFAULT_SHIFTS[0])
    return ch1[..., np.newaxis]  # (224, 224, 1)


def generate_multi_channel(iq_samples):
    """Generate multi-channel image (3-channel RGB)."""
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, DEFAULT_SHIFTS[0])
    ch2 = hardware_gen_layer(iq_samples, KERNEL_MEDIUM, DEFAULT_SHIFTS[1])
    ch3 = hardware_gen_layer(iq_samples, KERNEL_BLUR, DEFAULT_SHIFTS[2])
    return np.stack([ch1, ch2, ch3], axis=-1)  # (224, 224, 3)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_dataset(X, y_hdf5, mode='multi', verbose=True):
    """
    Generate image dataset from IQ samples.
    
    Args:
        X: IQ samples array (N, 1024, 2)
        y_hdf5: HDF5 class indices
        mode: 'single' or 'multi'
    
    Returns:
        images: (N, 224, 224, C) where C=1 or 3
        labels: (N,) model class indices
    """
    gen_func = generate_single_channel if mode == 'single' else generate_multi_channel
    
    images = []
    labels = []
    
    iterator = tqdm(range(len(X)), desc=f"Generating {mode}-channel") if verbose else range(len(X))
    
    for i in iterator:
        img = gen_func(X[i])
        label = HDF5_TO_MODEL_MAP[y_hdf5[i]]
        images.append(img)
        labels.append(label)
    
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels)


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_squeezenet_simple(input_shape, num_classes):
    """
    Simplified SqueezeNet for comparison.
    Handles both (224,224,1) and (224,224,3) inputs.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Fire modules (simplified)
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
    
    # Classifier
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), padding='same')(x)  # No activation before softmax
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax')(x)
    
    return keras.Model(inputs, outputs)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs=10, batch_size=32):
    """Train SqueezeNet model and return history."""
    model = build_squeezenet_simple(input_shape, NUM_CLASSES)
    
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print("=" * 80)
    print("PHASE 2: SINGLE vs MULTI-CHANNEL ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    
    # Settings
    SUBSET_FRACTION = 0.10  # 10% of data
    EPOCHS = 10
    BATCH_SIZE = 64
    VAL_SPLIT = 0.2
    
    print(f"\nConfiguration:")
    print(f"  Data subset: {SUBSET_FRACTION*100:.0f}%")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Validation split: {VAL_SPLIT*100:.0f}%")
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    X, Y, Z = load_data_sample(data_path)
    
    y_int = np.argmax(Y, axis=1) if Y.ndim > 1 else Y.flatten()
    
    # Filter to valid classes only
    mask = np.isin(y_int, VALID_HDF5_CLASSES)
    X = X[mask]
    y_int = y_int[mask]
    Z = Z[mask]
    
    print(f"  Total valid samples: {len(X)}")
    
    # Take subset
    n_subset = int(len(X) * SUBSET_FRACTION)
    indices = np.random.permutation(len(X))[:n_subset]
    X_subset = X[indices]
    y_subset = y_int[indices]
    
    print(f"  Using subset: {n_subset} samples")
    
    # Split train/val
    n_val = int(n_subset * VAL_SPLIT)
    n_train = n_subset - n_val
    
    X_train_iq = X_subset[:n_train]
    y_train_hdf5 = y_subset[:n_train]
    X_val_iq = X_subset[n_train:]
    y_val_hdf5 = y_subset[n_train:]
    
    print(f"  Train: {n_train}, Val: {n_val}")
    
    # Results storage
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_local/phase2_comparison/{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # =========================================================================
    # EXPERIMENT A: SINGLE-CHANNEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT A: SINGLE-CHANNEL (224x224x1)")
    print("=" * 80)
    
    print("\nGenerating single-channel images...")
    X_train_single, y_train = generate_dataset(X_train_iq, y_train_hdf5, mode='single')
    X_val_single, y_val = generate_dataset(X_val_iq, y_val_hdf5, mode='single', verbose=False)
    
    print(f"  Train shape: {X_train_single.shape}")
    print(f"  Val shape: {X_val_single.shape}")
    
    print("\nTraining single-channel model...")
    model_single, history_single = train_model(
        X_train_single, y_train,
        X_val_single, y_val,
        input_shape=(GRID_SIZE, GRID_SIZE, 1),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    single_val_acc = history_single.history['val_accuracy'][-1]
    single_train_acc = history_single.history['accuracy'][-1]
    
    results['single_channel'] = {
        'val_accuracy': float(single_val_acc),
        'train_accuracy': float(single_train_acc),
        'input_shape': [GRID_SIZE, GRID_SIZE, 1],
        'params': model_single.count_params(),
        'history': {
            'accuracy': [float(x) for x in history_single.history['accuracy']],
            'val_accuracy': [float(x) for x in history_single.history['val_accuracy']],
        }
    }
    
    print(f"\n>>> SINGLE-CHANNEL Results:")
    print(f"    Train Accuracy: {single_train_acc*100:.2f}%")
    print(f"    Val Accuracy: {single_val_acc*100:.2f}%")
    print(f"    Model Params: {model_single.count_params():,}")
    
    # Save model
    model_single.save(f'{results_dir}/model_single_channel.h5')
    
    # Free memory
    del X_train_single, X_val_single, model_single
    tf.keras.backend.clear_session()
    
    # =========================================================================
    # EXPERIMENT B: MULTI-CHANNEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT B: MULTI-CHANNEL (224x224x3)")
    print("=" * 80)
    
    print("\nGenerating multi-channel images...")
    X_train_multi, y_train = generate_dataset(X_train_iq, y_train_hdf5, mode='multi')
    X_val_multi, y_val = generate_dataset(X_val_iq, y_val_hdf5, mode='multi', verbose=False)
    
    print(f"  Train shape: {X_train_multi.shape}")
    print(f"  Val shape: {X_val_multi.shape}")
    
    print("\nTraining multi-channel model...")
    model_multi, history_multi = train_model(
        X_train_multi, y_train,
        X_val_multi, y_val,
        input_shape=(GRID_SIZE, GRID_SIZE, 3),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    multi_val_acc = history_multi.history['val_accuracy'][-1]
    multi_train_acc = history_multi.history['accuracy'][-1]
    
    results['multi_channel'] = {
        'val_accuracy': float(multi_val_acc),
        'train_accuracy': float(multi_train_acc),
        'input_shape': [GRID_SIZE, GRID_SIZE, 3],
        'params': model_multi.count_params(),
        'history': {
            'accuracy': [float(x) for x in history_multi.history['accuracy']],
            'val_accuracy': [float(x) for x in history_multi.history['val_accuracy']],
        }
    }
    
    print(f"\n>>> MULTI-CHANNEL Results:")
    print(f"    Train Accuracy: {multi_train_acc*100:.2f}%")
    print(f"    Val Accuracy: {multi_val_acc*100:.2f}%")
    print(f"    Model Params: {model_multi.count_params():,}")
    
    # Save model
    model_multi.save(f'{results_dir}/model_multi_channel.h5')
    
    # =========================================================================
    # COMPARISON & RECOMMENDATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    acc_diff = multi_val_acc - single_val_acc
    
    print(f"\n{'Metric':<25} {'Single (1ch)':<15} {'Multi (3ch)':<15} {'Diff':<10}")
    print("-" * 65)
    print(f"{'Val Accuracy':<25} {single_val_acc*100:>12.2f}% {multi_val_acc*100:>12.2f}% {acc_diff*100:>+8.2f}%")
    print(f"{'Train Accuracy':<25} {single_train_acc*100:>12.2f}% {multi_train_acc*100:>12.2f}%")
    print(f"{'Model Params':<25} {results['single_channel']['params']:>12,} {results['multi_channel']['params']:>12,}")
    print(f"{'Image Gen Compute':<25} {'1x':>12} {'3x':>12}")
    print(f"{'Memory Bandwidth':<25} {'1x':>12} {'3x':>12}")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    threshold = 0.05  # 5% threshold
    
    if acc_diff <= threshold:
        recommendation = "SINGLE-CHANNEL"
        reason = f"Multi-channel advantage ({acc_diff*100:.1f}%) is within {threshold*100:.0f}% threshold"
    else:
        recommendation = "MULTI-CHANNEL"
        reason = f"Multi-channel advantage ({acc_diff*100:.1f}%) exceeds {threshold*100:.0f}% threshold"
    
    results['recommendation'] = {
        'choice': recommendation,
        'reason': reason,
        'accuracy_difference': float(acc_diff),
        'threshold': threshold,
    }
    
    print(f"\n  >>> RECOMMENDED: {recommendation}")
    print(f"  >>> Reason: {reason}")
    
    if recommendation == "SINGLE-CHANNEL":
        print("\n  FPGA Benefits:")
        print("    - 1/3 image generation compute")
        print("    - 1/3 memory bandwidth")
        print("    - Simpler pipeline (1 kernel instead of 3)")
        print("    - Potential for smaller/faster model")
    
    # Save results
    results['config'] = {
        'subset_fraction': SUBSET_FRACTION,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'val_split': VAL_SPLIT,
        'n_train': n_train,
        'n_val': n_val,
        'timestamp': timestamp,
    }
    
    with open(f'{results_dir}/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}/")
    
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print(f"  1. Review results in {results_dir}/comparison_results.json")
    print(f"  2. Based on recommendation, proceed to Phase 3:")
    print(f"     - Generate full training dataset ({recommendation.lower()})")
    print(f"     - Retrain SqueezeNet from scratch")
    print(f"     - Validate with hardware simulation")


if __name__ == "__main__":
    main()
