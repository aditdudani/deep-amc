"""
Phase 2: Architecture Matrix - Comprehensive Single/Dual/Triple Channel Comparison

Tests 6 configurations selected from Phase 1 Grid Search results:

TIER 1: Single-Channel
  Config A: K02 (3x3 Cross)  - Efficiency Champion (69.35%, cost=5)
  Config B: K09 (7x7 Box)    - Accuracy Champion (69.84%, cost=49)
  Config C: K12 (11x11 Box)  - Low-SNR Integrator (0dB champion, cost=121)

TIER 2: Dual-Channel
  Config D: K02 + K09        - Mid-Range Pairing (cost=54)
  Config E: K02 + K12        - Extremes Pairing (cost=126)

TIER 3: Triple-Channel
  Config F: K02 + K09 + K12  - Software Proxy (cost=175)

Execution: Full dataset, 40 epochs, rolling disk generation.
"""

import os
import sys
import gc
import json
import shutil
import argparse
from datetime import datetime

# Suppress TensorFlow logging before import
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
            return
        # Skip lines that look like progress bars
        if any(pattern in message for pattern in ['[====', 'ETA:', '━', 'it/s]', '- loss:']):
            return
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
GAIN = 128

# Dataset parameters
DATA_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]
HDF5_TO_MODEL_MAP = {1: 2, 3: 5, 4: 7, 5: 4, 12: 0, 13: 1, 14: 3, 23: 6}
HDF5_CLASS_NAMES = {
    1: '4ASK', 3: 'BPSK', 4: 'QPSK', 5: '8PSK',
    12: '16QAM', 13: '32QAM', 14: '64QAM', 23: 'OQPSK'
}
MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
NUM_CLASSES = 8
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# Training config - FULL TRAINING
TRAIN_VAL_SPLIT = 0.90      # 90/10 train/val
EPOCHS = 40                  # Full training
BATCH_SIZE = 64
LEARNING_RATE = 0.01         # SGD with momentum
PATIENCE = 3                 # ReduceLROnPlateau patience (decreased from 5)
RANDOM_SEED = 42

# Output directories
TEMP_DATA_DIR = 'data/phase2_temp'
RESULTS_DIR = 'results_local/phase2_matrix'


# =============================================================================
# KERNEL DEFINITIONS - From Phase 1 Grid Search Champions
# =============================================================================

def create_cross_kernel(size):
    """Cross/Manhattan kernel - center row + column only."""
    kernel = np.zeros((size, size), dtype=np.int32)
    center = size // 2
    kernel[center, :] = GAIN
    kernel[:, center] = GAIN
    return kernel

def create_box_kernel(size):
    """Uniform/Box kernel - all ones."""
    return np.ones((size, size), dtype=np.int32) * GAIN

# The 3 champion kernels from Phase 1
KERNEL_K02 = create_cross_kernel(3)   # 3x3 Cross - Efficiency Champion
KERNEL_K09 = create_box_kernel(7)     # 7x7 Box - Accuracy Champion
KERNEL_K12 = create_box_kernel(11)    # 11x11 Box - 0dB Integrator

# Empirically calibrated shifts (from kernel_grid_search.py fix)
SHIFT_K02 = 0   # 3x3: no shift needed
SHIFT_K09 = 2   # 7x7: moderate shift
SHIFT_K12 = 3   # 11x11: larger shift


# =============================================================================
# 6-CONFIGURATION MATRIX
# =============================================================================

def define_configurations():
    """
    Define the 6 configurations for Phase 2 Architecture Matrix.
    Each config specifies: kernels, shifts, num_channels, cost, description.
    """
    configs = {
        'A': {
            'name': 'K02_3x3Cross',
            'tier': 'Single',
            'kernels': [KERNEL_K02],
            'shifts': [SHIFT_K02],
            'num_channels': 1,
            'cost': 5,
            'description': 'Efficiency Champion - Best accuracy/cost ratio',
        },
        'B': {
            'name': 'K09_7x7Box',
            'tier': 'Single',
            'kernels': [KERNEL_K09],
            'shifts': [SHIFT_K09],
            'num_channels': 1,
            'cost': 49,
            'description': 'Accuracy Champion - Highest Phase 1 accuracy',
        },
        'C': {
            'name': 'K12_11x11Box',
            'tier': 'Single',
            'kernels': [KERNEL_K12],
            'shifts': [SHIFT_K12],
            'num_channels': 1,
            'cost': 121,
            'description': '0dB Integrator - Best low-SNR performance',
        },
        'D': {
            'name': 'K02+K09',
            'tier': 'Dual',
            'kernels': [KERNEL_K02, KERNEL_K09],
            'shifts': [SHIFT_K02, SHIFT_K09],
            'num_channels': 2,
            'cost': 5 + 49,
            'description': 'Mid-Range Pairing - High-SNR + Overall champion',
        },
        'E': {
            'name': 'K02+K12',
            'tier': 'Dual',
            'kernels': [KERNEL_K02, KERNEL_K12],
            'shifts': [SHIFT_K02, SHIFT_K12],
            'num_channels': 2,
            'cost': 5 + 121,
            'description': 'Extremes Pairing - Sharp + Heavy integration',
        },
        'F': {
            'name': 'K02+K09+K12',
            'tier': 'Triple',
            'kernels': [KERNEL_K02, KERNEL_K09, KERNEL_K12],
            'shifts': [SHIFT_K02, SHIFT_K09, SHIFT_K12],
            'num_channels': 3,
            'cost': 5 + 49 + 121,
            'description': 'Software Proxy - Full multi-scale representation',
        },
    }
    return configs


# =============================================================================
# HARDWARE IMAGE GENERATION
# =============================================================================

def hw_gen_layer(iq_samples, kernel, shift_val):
    """Hardware-accurate image generation with integer-only arithmetic."""
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
    
    # Simulate FPGA int16 clamping
    accumulator = np.clip(accumulator, -32768, 32767)
    
    output = accumulator >> shift_val
    return np.clip(output, 0, 255).astype(np.uint8)


def generate_image(iq_samples, kernels, shifts):
    """
    Generate image with 1, 2, or 3 channels based on config.
    Always returns (224, 224, 3) for model compatibility.
    """
    channels = []
    for kernel, shift in zip(kernels, shifts):
        ch = hw_gen_layer(iq_samples, kernel, shift)
        channels.append(ch)
    
    # Pad to 3 channels if needed
    if len(channels) == 1:
        # Single channel: replicate to RGB
        return np.stack([channels[0], channels[0], channels[0]], axis=-1)
    elif len(channels) == 2:
        # Dual channel: duplicate first channel as third
        return np.stack([channels[0], channels[1], channels[0]], axis=-1)
    else:
        # Triple channel: direct RGB
        return np.stack(channels, axis=-1)


# =============================================================================
# DATASET GENERATION (Rolling Disk)
# =============================================================================

def generate_dataset_to_disk(config, output_dir, seed=RANDOM_SEED):
    """
    Generate FULL dataset for a configuration.
    Uses stratified sampling across (class, SNR) combinations.
    """
    print(f"\n  Generating images to {output_dir}...")
    
    # Clear output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create directories for each class
    for split in ['train', 'validation']:
        for class_name in MODEL_CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    kernels = config['kernels']
    shifts = config['shifts']
    
    np.random.seed(seed)
    image_counts = {'train': 0, 'validation': 0}
    
    with h5py.File(DATA_PATH, 'r') as hf:
        X = hf['X']
        n_total = X.shape[0]
        
        # Preload labels and SNR to memory (avoid repeated disk reads)
        print(f"  Loading labels...")
        Y_onehot = hf['Y'][:]
        y_int = np.argmax(Y_onehot, axis=1)
        Z = hf['Z'][:]
        snr_values = Z[:, 0] if len(Z.shape) > 1 else Z
        del Y_onehot  # Free memory
        
        print(f"  Processing {n_total:,} total samples...")
        
        # Process by (class, SNR) combinations for stratification
        for hdf5_class in tqdm(VALID_HDF5_CLASSES, desc="  Classes"):
            class_name = HDF5_CLASS_NAMES[hdf5_class]
            class_mask = (y_int == hdf5_class)
            
            for snr in TARGET_SNRS:
                # Find samples matching this class and SNR
                snr_mask = (snr_values == snr)
                combo_mask = class_mask & snr_mask
                combo_indices = np.where(combo_mask)[0]
                
                if len(combo_indices) == 0:
                    continue
                
                # Shuffle indices for random split
                np.random.shuffle(combo_indices)
                
                # Split into train/val
                n_samples = len(combo_indices)
                n_train = int(n_samples * TRAIN_VAL_SPLIT)
                
                train_indices = combo_indices[:n_train]
                val_indices = combo_indices[n_train:]
                
                # Generate and save images
                for split, indices in [('train', train_indices), ('validation', val_indices)]:
                    for idx in indices:
                        iq = X[idx]
                        img = generate_image(iq, kernels, shifts)
                        
                        fname = f"snr{snr:+03d}_idx{idx:07d}.png"
                        fpath = os.path.join(output_dir, split, class_name, fname)
                        
                        from PIL import Image
                        Image.fromarray(img).save(fpath)
                        image_counts[split] += 1
    
    print(f"  Generated {image_counts['train']:,} train, {image_counts['validation']:,} val images")
    return image_counts


# =============================================================================
# SQUEEZENET MODEL
# =============================================================================

def _fire_module(x, squeeze_channels, expand_channels):
    squeeze = layers.Conv2D(squeeze_channels, (1, 1), activation='relu', padding='valid',
                            kernel_initializer='he_normal')(x)
    expand_1x1 = layers.Conv2D(expand_channels, (1, 1), activation='relu', padding='valid',
                               kernel_initializer='he_normal')(squeeze)
    expand_3x3 = layers.Conv2D(expand_channels, (3, 3), activation='relu', padding='same',
                               kernel_initializer='he_normal')(squeeze)
    return layers.Concatenate(axis=-1)([expand_1x1, expand_3x3])


def build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=8, dropout_rate=0.5):
    """SqueezeNet v1.1 architecture."""
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
    x = layers.Conv2D(num_classes, (1, 1), activation=None, padding='valid',
                      name='conv_final', dtype='float32', kernel_initializer='he_normal')(x)
    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    outputs = layers.Softmax(name='predictions')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='squeezenet_v1_1')


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


def train_and_evaluate(config_id, config, output_dir, results_dir):
    """
    Train SqueezeNet and evaluate per-SNR accuracy.
    Returns dict with overall and per-SNR metrics.
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    
    print(f"\n  Training SqueezeNet for {EPOCHS} epochs...")
    
    # Create datasets
    train_ds, class_names = make_dataset(train_dir, BATCH_SIZE, shuffle=True)
    val_ds, _ = make_dataset(val_dir, BATCH_SIZE, shuffle=False)
    
    # Build model
    tf.keras.backend.clear_session()
    model = build_squeezenet_v11(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES,
        dropout_rate=0.5
    )
    
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    # Callbacks
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=PATIENCE,  # Decreased to 3
        min_lr=1e-6,
        verbose=1
    )
    
    # Model checkpoint - save best weights
    model_path = os.path.join(results_dir, f'model_config_{config_id}.keras')
    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )
    
    # Get metrics
    overall_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n  Overall val_acc: {overall_val_acc*100:.2f}% (best: {best_val_acc*100:.2f}%)")
    
    # Load best model for evaluation
    model = keras.models.load_model(model_path)
    
    # Per-SNR evaluation
    print(f"  Evaluating per-SNR accuracy...")
    snr_accuracies = evaluate_per_snr(model, val_dir, class_names)
    
    results = {
        'config_id': config_id,
        'name': config['name'],
        'tier': config['tier'],
        'num_channels': config['num_channels'],
        'cost': config['cost'],
        'description': config['description'],
        'overall_val_acc': float(best_val_acc),
        'final_val_acc': float(overall_val_acc),
        'final_val_loss': float(final_val_loss),
        'per_snr_accuracy': snr_accuracies,
        'model_path': model_path,
    }
    
    # Print per-SNR results
    print(f"  Per-SNR accuracy:")
    for snr in TARGET_SNRS:
        acc = snr_accuracies.get(str(snr), 0)
        print(f"    SNR {snr:>3}dB: {acc*100:.1f}%")
    
    # Cleanup model from memory
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    return results


def evaluate_per_snr(model, val_dir, class_names):
    """Evaluate model accuracy for each SNR level (batched for efficiency)."""
    snr_correct = {snr: 0 for snr in TARGET_SNRS}
    snr_total = {snr: 0 for snr in TARGET_SNRS}
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Collect all images with metadata
    all_images = []
    
    for class_name in MODEL_CLASS_NAMES:
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        true_label = class_to_idx.get(class_name)
        if true_label is None:
            continue
        
        for fname in os.listdir(class_dir):
            if not fname.endswith('.png'):
                continue
            
            try:
                snr_str = fname.split('_')[0]
                snr_val = int(snr_str[3:])
            except (ValueError, IndexError):
                continue
            
            if snr_val not in TARGET_SNRS:
                continue
            
            img_path = os.path.join(class_dir, fname)
            all_images.append((img_path, true_label, snr_val))
    
    # Process in batches
    batch_size = 64
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i+batch_size]
        
        images = []
        for img_path, _, _ in batch:
            img = tf.keras.utils.load_img(img_path, target_size=(GRID_SIZE, GRID_SIZE))
            img_array = tf.keras.utils.img_to_array(img)
            images.append(img_array)
        
        images = np.stack(images, axis=0)
        predictions = model.predict(images, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        for j, (_, true_label, snr_val) in enumerate(batch):
            snr_total[snr_val] += 1
            if pred_labels[j] == true_label:
                snr_correct[snr_val] += 1
    
    # Calculate accuracies
    snr_accuracies = {}
    for snr in TARGET_SNRS:
        if snr_total[snr] > 0:
            snr_accuracies[str(snr)] = snr_correct[snr] / snr_total[snr]
        else:
            snr_accuracies[str(snr)] = 0.0
    
    return snr_accuracies


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_results(all_results):
    """Comprehensive analysis of all 6 configurations."""
    print("\n" + "=" * 80)
    print("PHASE 2 ARCHITECTURE MATRIX - RESULTS ANALYSIS")
    print("=" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x['overall_val_acc'], reverse=True)
    
    # Overall ranking
    print("\n" + "-" * 60)
    print("OVERALL ACCURACY RANKING")
    print("-" * 60)
    print(f"{'Rank':<5} {'Config':<6} {'Name':<18} {'Tier':<8} {'Accuracy':<10} {'Cost':<8}")
    print("-" * 60)
    for i, r in enumerate(sorted_results, 1):
        print(f"{i:<5} {r['config_id']:<6} {r['name']:<18} {r['tier']:<8} {r['overall_val_acc']*100:>7.2f}%  {r['cost']:<8}")
    
    # Tier comparison
    print("\n" + "-" * 60)
    print("TIER COMPARISON")
    print("-" * 60)
    
    tiers = {'Single': [], 'Dual': [], 'Triple': []}
    for r in all_results:
        tiers[r['tier']].append(r)
    
    for tier_name, tier_results in tiers.items():
        if tier_results:
            best = max(tier_results, key=lambda x: x['overall_val_acc'])
            avg = np.mean([r['overall_val_acc'] for r in tier_results])
            print(f"{tier_name}-Channel: Best={best['overall_val_acc']*100:.2f}% ({best['name']}), Avg={avg*100:.2f}%")
    
    # Per-SNR analysis
    print("\n" + "-" * 60)
    print("PER-SNR CHAMPIONS")
    print("-" * 60)
    
    for snr in TARGET_SNRS:
        snr_key = str(snr)
        best_config = max(all_results, key=lambda x: x['per_snr_accuracy'].get(snr_key, 0))
        best_acc = best_config['per_snr_accuracy'].get(snr_key, 0)
        print(f"SNR {snr:>3}dB: {best_config['name']:<18} ({best_acc*100:.1f}%)")
    
    # Single vs Multi decision
    print("\n" + "-" * 60)
    print("ARCHITECTURE DECISION")
    print("-" * 60)
    
    best_single = max([r for r in all_results if r['tier'] == 'Single'], 
                      key=lambda x: x['overall_val_acc'])
    best_multi = max([r for r in all_results if r['tier'] != 'Single'], 
                     key=lambda x: x['overall_val_acc'])
    
    single_acc = best_single['overall_val_acc']
    multi_acc = best_multi['overall_val_acc']
    acc_diff = multi_acc - single_acc
    
    print(f"Best Single-Channel: {best_single['name']} ({single_acc*100:.2f}%, cost={best_single['cost']})")
    print(f"Best Multi-Channel:  {best_multi['name']} ({multi_acc*100:.2f}%, cost={best_multi['cost']})")
    print(f"Accuracy Difference: {acc_diff*100:+.2f}%")
    
    threshold = 0.05  # 5%
    if acc_diff <= threshold:
        recommendation = best_single['name']
        reason = f"Multi-channel gain ({acc_diff*100:.2f}%) ≤ {threshold*100:.0f}% threshold"
        fpga_benefit = "Single-channel saves bandwidth and complexity"
    else:
        recommendation = best_multi['name']
        reason = f"Multi-channel gain ({acc_diff*100:.2f}%) > {threshold*100:.0f}% threshold"
        fpga_benefit = f"Multi-channel justified by {acc_diff*100:.2f}% accuracy gain"
    
    print(f"\n>>> RECOMMENDATION: {recommendation}")
    print(f">>> Reason: {reason}")
    print(f">>> FPGA: {fpga_benefit}")
    
    return {
        'recommendation': recommendation,
        'best_single': best_single['name'],
        'best_multi': best_multi['name'],
        'accuracy_difference': acc_diff,
    }


def save_results(all_results, output_path):
    """Save results to JSON and CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # JSON (full details)
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {json_path}")
    
    # CSV (for easy analysis)
    csv_lines = ['config_id,name,tier,channels,cost,overall_acc,best_acc,' + 
                 ','.join([f'snr_{s}dB' for s in TARGET_SNRS])]
    
    for r in all_results:
        snr_accs = [f"{r['per_snr_accuracy'].get(str(s), 0)*100:.2f}" for s in TARGET_SNRS]
        line = f"{r['config_id']},{r['name']},{r['tier']},{r['num_channels']},{r['cost']},"
        line += f"{r['overall_val_acc']*100:.2f},{r['final_val_acc']*100:.2f},"
        line += ','.join(snr_accs)
        csv_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"Saved rankings to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2 Architecture Matrix')
    parser.add_argument('--config', type=str, help='Test single config only (e.g., A, B, C, D, E, F)')
    parser.add_argument('--resume', type=str, help='Resume from specific config')
    parser.add_argument('--no-log', action='store_true', help='Disable clean log file')
    args = parser.parse_args()
    
    # Set up clean logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = None
    if not args.no_log:
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/phase2_matrix_{timestamp}.log'
        logger = TeeLogger(log_path)
        sys.stdout = logger
        print(f"Clean log: {log_path}")
    
    print("\n" + "=" * 80)
    print("PHASE 2: ARCHITECTURE MATRIX")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {DATA_PATH} (FULL DATASET)")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Optimizer: SGD(lr={LEARNING_RATE}, momentum=0.9)")
    print(f"ReduceLROnPlateau: patience={PATIENCE}")
    print("=" * 80)
    
    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    
    # Define configurations
    configs = define_configurations()
    
    # Print configuration matrix
    print("\n6-CONFIGURATION MATRIX:")
    print("-" * 80)
    for cid, cfg in configs.items():
        print(f"  Config {cid}: {cfg['name']:<20} | {cfg['tier']:<7} | Cost={cfg['cost']:<5} | {cfg['description']}")
    print("-" * 80)
    
    # Filter if single config requested
    if args.config:
        if args.config.upper() not in configs:
            print(f"ERROR: Unknown config {args.config}. Valid: A, B, C, D, E, F")
            return
        configs = {args.config.upper(): configs[args.config.upper()]}
    
    # Resume handling
    start_idx = 0
    config_ids = list(configs.keys())
    if args.resume:
        if args.resume.upper() in config_ids:
            start_idx = config_ids.index(args.resume.upper())
            print(f"\nResuming from Config {args.resume.upper()} (index {start_idx})")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = []
    
    # Main loop
    for i, (config_id, config) in enumerate(list(configs.items())[start_idx:], start_idx + 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(configs)}] TESTING CONFIG {config_id}: {config['name']}")
        print(f"  Tier: {config['tier']}, Channels: {config['num_channels']}, Cost: {config['cost']}")
        print(f"  {config['description']}")
        print("=" * 80)
        
        # Step 1: Generate dataset
        generate_dataset_to_disk(config, TEMP_DATA_DIR, seed=RANDOM_SEED)
        
        # Step 2: Train and evaluate
        results = train_and_evaluate(config_id, config, TEMP_DATA_DIR, RESULTS_DIR)
        all_results.append(results)
        
        # Step 3: Save intermediate results
        save_results(all_results, os.path.join(RESULTS_DIR, 'phase2_rankings.csv'))
        
        # Step 4: Clean up temp directory
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)
            print(f"  Cleaned up temp directory")
    
    # Final analysis
    decision = analyze_results(all_results)
    
    # Save final results with timestamp
    final_path = os.path.join(RESULTS_DIR, f'phase2_rankings_{timestamp}.csv')
    save_results(all_results, final_path)
    
    # Add decision to results
    final_json = os.path.join(RESULTS_DIR, f'phase2_final_{timestamp}.json')
    final_data = {
        'timestamp': timestamp,
        'configs': all_results,
        'decision': decision,
        'parameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'patience': PATIENCE,
            'dropout': 0.5,
        }
    }
    with open(final_json, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"\nSaved final analysis to {final_json}")
    
    print("\n" + "=" * 80)
    print("PHASE 2 ARCHITECTURE MATRIX COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Best models saved with weights.")
    print("=" * 80)
    
    # Close logger
    if logger:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
