"""
Phase 2: Architecture Decision - Single vs Multi-Channel Comparison

RIGOROUS COMPARISON for deterministic architecture decision.

Strategy:
1. Generate hardware images to disk (data/processed_hw_single/ and data/processed_hw_multi/)
2. Train using image_dataset_from_directory (streams from disk, no OOM)
3. Multiple training runs with different seeds for statistical confidence
4. Compare accuracy to decide optimal architecture

Decision Rule: If single-channel is within 5% of multi-channel → go single for FPGA savings.
"""

import os
import sys

# Suppress TensorFlow logging before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TF warnings

from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
import json
import shutil
from datetime import datetime
from tqdm import tqdm
import h5py

# GPU setup (silent except for count)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
print(f"GPUs available: {len(gpus)}")

# Progress bar control: set NOPROGRESS=1 for clean logs
SHOW_PROGRESS = os.environ.get('NOPROGRESS', '0') != '1'

# Skip image generation if already exists: set SKIP_GEN=1 to reuse existing images
SKIP_GENERATION = os.environ.get('SKIP_GEN', '0') == '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# HARDWARE PARAMETERS (MUST MATCH simulate_hardware.py / calibrate_hardware.py)
# =============================================================================
GRID_SIZE = 224
GAIN = 128

# Kernels - EXACTLY as calibrated in Phase 1
KERNEL_SHARP = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.int16) * GAIN

KERNEL_MEDIUM = np.array([[1, 2, 1],
                          [2, 8, 2],
                          [1, 2, 1]], dtype=np.int16) * GAIN

KERNEL_BLUR = np.ones((11, 11), dtype=np.int16) * GAIN

# Calibrated shifts from Phase 1
DEFAULT_SHIFTS = (0, 0, 3)

# =============================================================================
# DATASET PARAMETERS (from RadioML GOLD_XYZ_OSC.0001_1024.hdf5)
# =============================================================================
# HDF5 structure:
#   X: shape (2555904, 1024, 2) - I/Q samples
#   Y: shape (2555904, 24) - one-hot encoded modulation labels
#   Z: shape (2555904, 1) - SNR values (need [:, 0] or .flatten())
#
# 24 modulations total, we use 8 for AMC classification:

VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]  # Indices in one-hot Y
HDF5_TO_MODEL_MAP = {1: 2, 3: 5, 4: 7, 5: 4, 12: 0, 13: 1, 14: 3, 23: 6}
HDF5_CLASS_NAMES = {
    1: '4ASK',    # Index 1 in one-hot → model class 2
    3: 'BPSK',    # Index 3 in one-hot → model class 5
    4: 'QPSK',    # Index 4 in one-hot → model class 7
    5: '8PSK',    # Index 5 in one-hot → model class 4
    12: '16QAM',  # Index 12 in one-hot → model class 0
    13: '32QAM',  # Index 13 in one-hot → model class 1
    14: '64QAM',  # Index 14 in one-hot → model class 3
    23: 'OQPSK',  # Index 23 in one-hot → model class 6
}
MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
NUM_CLASSES = 8

# Target SNRs (matching original preprocessing - 0 to 10 dB in 2dB steps)
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# =============================================================================
# EXPERIMENT CONFIGURATION - RIGOROUS SETTINGS
# =============================================================================
class ExperimentConfig:
    """Configuration for rigorous, deterministic comparison."""
    
    # Data settings
    DATA_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    SUBSET_FRACTION = 0.30      # 30% of data for robust statistics (was 10%)
    TRAIN_VAL_SPLIT = 0.90      # 90% train, 10% validation
    
    # Training settings
    EPOCHS = 40                 # Matching original train_squeezenet.py
    BATCH_SIZE = 64             # Matching original
    BASE_LEARNING_RATE = 1e-2   # Starting LR (ReduceLROnPlateau will adjust)
    MIN_LEARNING_RATE = 1e-6    # Minimum LR for scheduler
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    # Single seed per mode (2 total runs)
    RANDOM_SEED = 42
    
    # Decision threshold
    ACCURACY_THRESHOLD = 0.05   # 5% threshold for architecture decision
    
    @classmethod
    def print_config(cls):
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"  Data subset:         {cls.SUBSET_FRACTION*100:.0f}%")
        print(f"  Train/Val split:     {cls.TRAIN_VAL_SPLIT*100:.0f}/{(1-cls.TRAIN_VAL_SPLIT)*100:.0f}")
        print(f"  Epochs:              {cls.EPOCHS}")
        print(f"  Batch size:          {cls.BATCH_SIZE}")
        print(f"  Base learning rate:  {cls.BASE_LEARNING_RATE}")
        print(f"  Early stop patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"  Random seed:         {cls.RANDOM_SEED}")
        print(f"  Decision threshold:  {cls.ACCURACY_THRESHOLD*100:.0f}%")
        print("=" * 60)


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


def generate_single_channel_image(iq_samples):
    """Generate single-channel image (Ch1 sharp only) - returns (224, 224, 1)."""
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, DEFAULT_SHIFTS[0])
    # Stack 3 times to make RGB for compatibility with standard models
    return np.stack([ch1, ch1, ch1], axis=-1)


def generate_multi_channel_image(iq_samples):
    """Generate multi-channel image (3-channel RGB) - returns (224, 224, 3)."""
    ch1 = hardware_gen_layer(iq_samples, KERNEL_SHARP, DEFAULT_SHIFTS[0])
    ch2 = hardware_gen_layer(iq_samples, KERNEL_MEDIUM, DEFAULT_SHIFTS[1])
    ch3 = hardware_gen_layer(iq_samples, KERNEL_BLUR, DEFAULT_SHIFTS[2])
    return np.stack([ch1, ch2, ch3], axis=-1)


# =============================================================================
# DATASET GENERATION (to disk)
# =============================================================================

def generate_dataset_to_disk(data_path, output_dir, mode='multi', subset_fraction=0.30, 
                              train_val_split=0.9, seed=42):
    """
    Generate hardware images and save to disk in folder structure compatible with
    image_dataset_from_directory.
    
    STRATIFIED SAMPLING: Equal samples from each (class, SNR) combination to ensure
    balanced representation across all SNR levels.
    
    HDF5 Structure (verified from dataset.txt):
        X: shape (2555904, 1024, 2) - I/Q samples as float32
        Y: shape (2555904, 24) - one-hot encoded modulation
        Z: shape (2555904, 1) - SNR values (need [:, 0] to get 1D array)
    
    Output structure:
        output_dir/
            train/
                16QAM/
                ...
            validation/
                16QAM/
                ...
    """
    # Check if we should skip generation (reuse existing images)
    if SKIP_GENERATION and os.path.exists(output_dir):
        train_dir = os.path.join(output_dir, 'train')
        if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
            print(f"\n{'='*60}")
            print(f"SKIPPING {mode.upper()}-CHANNEL GENERATION (SKIP_GEN=1)")
            print(f"Using existing images in: {output_dir}")
            print(f"{'='*60}")
            # Return dummy counts
            return {'train': 0, 'validation': 0}, {snr: {'train': 0, 'val': 0} for snr in TARGET_SNRS}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"GENERATING {mode.upper()}-CHANNEL DATASET")
    print(f"{'='*60}")
    
    gen_func = generate_single_channel_image if mode == 'single' else generate_multi_channel_image
    
    # Clear output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create directories for each class
    for split in ['train', 'validation']:
        for class_name in MODEL_CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Load data from HDF5
    print(f"Loading from {data_path}...")
    
    with h5py.File(data_path, 'r') as hf:
        X = hf['X']
        Y_onehot = hf['Y'][:]
        y_int = np.argmax(Y_onehot, axis=1)
        Z_2d = hf['Z'][:]
        snrs = (Z_2d[:, 0] if Z_2d.ndim > 1 else Z_2d.flatten()).astype(np.int32)
        
        print(f"  Total samples: {len(y_int)}")
        print(f"  Target classes: {[HDF5_CLASS_NAMES[c] for c in VALID_HDF5_CLASSES]}")
        print(f"  Target SNRs: {TARGET_SNRS} dB")
        
        # =====================================================================
        # STRATIFIED SAMPLING BY (CLASS, SNR) - ensures balanced SNR distribution
        # =====================================================================
        np.random.seed(seed)
        
        # Group indices by (class, snr) combination
        indices_by_class_snr = {}
        for cls in VALID_HDF5_CLASSES:
            for snr in TARGET_SNRS:
                key = (cls, snr)
                mask = (y_int == cls) & (snrs == snr)
                indices_by_class_snr[key] = np.where(mask)[0]
        
        # Calculate samples per (class, snr) combination
        # Total combinations = 8 classes × 6 SNRs = 48
        # Each combination has 4096 samples in the original dataset
        samples_per_combo = int(4096 * subset_fraction)  # e.g., 30% of 4096 = 1228
        
        print(f"\nStratified sampling:")
        print(f"  Samples per (class, SNR): {samples_per_combo}")
        print(f"  Total combinations: {len(VALID_HDF5_CLASSES)} classes × {len(TARGET_SNRS)} SNRs = {len(VALID_HDF5_CLASSES) * len(TARGET_SNRS)}")
        
        # Verify and collect stratified samples
        snr_counts = {snr: 0 for snr in TARGET_SNRS}
        class_counts = {cls: 0 for cls in VALID_HDF5_CLASSES}
        all_selected_indices = []
        
        print("\n  Per-SNR sample counts:")
        for snr in TARGET_SNRS:
            snr_total = 0
            for cls in VALID_HDF5_CLASSES:
                key = (cls, snr)
                available = indices_by_class_snr[key]
                if len(available) < samples_per_combo:
                    print(f"    WARNING: {HDF5_CLASS_NAMES[cls]} @ {snr}dB has only {len(available)} samples")
                n_take = min(samples_per_combo, len(available))
                selected = np.random.choice(available, n_take, replace=False)
                all_selected_indices.extend(selected)
                snr_counts[snr] += n_take
                class_counts[cls] += n_take
                snr_total += n_take
            print(f"    SNR {snr:>3}dB: {snr_total} samples")
        
        total_samples = len(all_selected_indices)
        print(f"\n  Total selected: {total_samples} samples")
        
        # Verify class balance
        print("\n  Per-class sample counts:")
        for cls in VALID_HDF5_CLASSES:
            print(f"    {HDF5_CLASS_NAMES[cls]:<8}: {class_counts[cls]}")
        
        # Convert to array and shuffle
        all_selected_indices = np.array(all_selected_indices)
        np.random.shuffle(all_selected_indices)
        
        # Group by class for train/val splitting
        indices_by_class = {cls: [] for cls in VALID_HDF5_CLASSES}
        for idx in all_selected_indices:
            indices_by_class[y_int[idx]].append(idx)
        
        # Generate images
        image_counts = {'train': 0, 'validation': 0}
        snr_image_counts = {snr: {'train': 0, 'val': 0} for snr in TARGET_SNRS}
        
        print("\nGenerating images...")
        for hdf5_cls, indices in indices_by_class.items():
            class_name = HDF5_CLASS_NAMES[hdf5_cls]
            
            if len(indices) == 0:
                continue
            
            # Shuffle and split
            np.random.seed(seed + hdf5_cls)
            indices = list(indices)
            np.random.shuffle(indices)
            split_point = int(len(indices) * train_val_split)
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]
            
            # Generate training images
            train_dir = os.path.join(output_dir, 'train', class_name)
            for i, idx in enumerate(tqdm(train_indices, desc=f"train/{class_name}", leave=False, disable=not SHOW_PROGRESS)):
                iq = np.asarray(X[idx], dtype=np.float32)
                img = gen_func(iq)
                snr_val = snrs[idx]
                fname = f"snr{snr_val:+03d}_idx{idx:07d}.png"
                tf.keras.utils.save_img(os.path.join(train_dir, fname), img)
                image_counts['train'] += 1
                snr_image_counts[snr_val]['train'] += 1
            
            # Generate validation images
            val_dir = os.path.join(output_dir, 'validation', class_name)
            for i, idx in enumerate(tqdm(val_indices, desc=f"val/{class_name}", leave=False, disable=not SHOW_PROGRESS)):
                iq = np.asarray(X[idx], dtype=np.float32)
                img = gen_func(iq)
                snr_val = snrs[idx]
                fname = f"snr{snr_val:+03d}_idx{idx:07d}.png"
                tf.keras.utils.save_img(os.path.join(val_dir, fname), img)
                image_counts['validation'] += 1
                snr_image_counts[snr_val]['val'] += 1
            
            print(f"  {class_name}: {len(train_indices)} train, {len(val_indices)} val")
        
        # Store SNR distribution for later
        snr_distribution = snr_image_counts
    
    print(f"\nGeneration complete: {image_counts['train']} train, {image_counts['validation']} val")
    
    # Verify SNR balance in final dataset
    print("\nSNR distribution in generated dataset:")
    for snr in TARGET_SNRS:
        print(f"  SNR {snr:>3}dB: {snr_distribution[snr]['train']} train, {snr_distribution[snr]['val']} val")
    
    # Save generation metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'timestamp': timestamp,
        'mode': mode,
        'seed': seed,
        'subset_fraction': subset_fraction,
        'train_val_split': train_val_split,
        'train_count': image_counts['train'],
        'val_count': image_counts['validation'],
        'target_snrs': TARGET_SNRS,
        'snr_distribution': {str(k): v for k, v in snr_distribution.items()},
        'classes': MODEL_CLASS_NAMES,
        'hardware_params': {
            'grid_size': GRID_SIZE,
            'gain': GAIN,
            'shifts': list(DEFAULT_SHIFTS),
        }
    }
    with open(os.path.join(output_dir, f'generation_metadata_{timestamp}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return image_counts, snr_distribution


# =============================================================================
# MODEL BUILDING (same as original train_squeezenet.py)
# =============================================================================

def build_squeezenet_v11(input_shape, num_classes, dropout_rate=0.0):
    """SqueezeNet v1.1 - same architecture as original training."""
    inputs = keras.Input(shape=input_shape)
    
    # Rescaling layer (0-255 -> 0-1)
    x = layers.Rescaling(1./255)(inputs)
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Fire modules
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
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax')(x)
    
    return keras.Model(inputs, outputs)


# =============================================================================
# LEARNING RATE PRINTER CALLBACK
# =============================================================================

class LearningRatePrinter(tf.keras.callbacks.Callback):
    """Print learning rate at the start of each epoch (minimal output)."""
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 10 == 0:  # Only print every 10 epochs
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print(f"[Epoch {epoch+1}] LR: {lr:.6g}")


# =============================================================================
# PER-SNR EVALUATION
# =============================================================================

def evaluate_per_snr(model, val_dir, class_names):
    """
    Evaluate model accuracy per SNR level.
    
    Filenames are formatted as: snr{snr:+03d}_idx{idx:07d}.png
    e.g., snr+00_idx0001234.png, snr+10_idx0005678.png
    
    Returns:
        dict mapping SNR -> accuracy
    """
    snr_results = {snr: {'correct': 0, 'total': 0} for snr in TARGET_SNRS}
    
    # Map class names to indices
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        true_label = class_to_idx[class_name]
        
        for fname in os.listdir(class_dir):
            if not fname.endswith('.png'):
                continue
            
            # Parse SNR from filename: snr+00_idx0001234.png
            try:
                snr_str = fname.split('_')[0]  # "snr+00"
                snr_val = int(snr_str[3:])  # Remove "snr" prefix, parse number
            except (ValueError, IndexError):
                continue
            
            if snr_val not in TARGET_SNRS:
                continue
            
            # Load and predict
            img_path = os.path.join(class_dir, fname)
            img = tf.keras.utils.load_img(img_path, target_size=(GRID_SIZE, GRID_SIZE))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = model.predict(img_array, verbose=0)
            pred_label = np.argmax(pred[0])
            
            snr_results[snr_val]['total'] += 1
            if pred_label == true_label:
                snr_results[snr_val]['correct'] += 1
    
    # Compute accuracy per SNR
    snr_accuracy = {}
    for snr in TARGET_SNRS:
        total = snr_results[snr]['total']
        correct = snr_results[snr]['correct']
        snr_accuracy[snr] = correct / total if total > 0 else 0.0
    
    return snr_accuracy


def make_datasets(train_dir, val_dir, image_size, batch_size):
    """Create tf.data datasets from directory."""
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Save class_names before prefetch (prefetch loses this attribute)
    class_names = train_ds.class_names
    
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def train_model(data_dir, results_dir, mode, seed, config):
    """
    Train SqueezeNet model with warmup and full callbacks.
    
    Returns:
        model, history, metrics dict
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{mode}_{timestamp}"
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {mode.upper()}-CHANNEL, Seed={seed}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}")
    
    # Set seeds for reproducibility
    tf.keras.utils.set_random_seed(seed)
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    
    train_ds, val_ds, class_names = make_datasets(train_dir, val_dir, GRID_SIZE, config.BATCH_SIZE)
    
    # Get class count and steps
    num_classes = len(class_names)
    steps_per_epoch = len(train_ds)
    
    print(f"  Classes: {class_names}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    model = build_squeezenet_v11(
        input_shape=(GRID_SIZE, GRID_SIZE, 3),
        num_classes=num_classes,
        dropout_rate=0.0
    )
    
    # Simple SGD with fixed LR (ReduceLROnPlateau callback will adjust)
    optimizer = optimizers.SGD(learning_rate=config.BASE_LEARNING_RATE, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    # Callbacks
    model_path = os.path.join(run_dir, f'model_best_{timestamp}.h5')
    
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max',
        verbose=1,
        restore_best_weights=True,
    )
    
    csv_logger = callbacks.CSVLogger(
        os.path.join(run_dir, f'training_log_{timestamp}.csv')
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=config.MIN_LEARNING_RATE,
        verbose=1,
    )
    
    callback_list = [
        checkpoint,
        early_stopping,
        csv_logger,
        reduce_lr,
        LearningRatePrinter(),
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callback_list,
        verbose=1,
    )
    
    # Collect metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    epochs_trained = len(history.history['accuracy'])
    
    metrics = {
        'mode': mode,
        'seed': seed,
        'timestamp': timestamp,
        'best_val_accuracy': float(best_val_acc),
        'best_epoch': best_epoch,
        'final_train_accuracy': float(final_train_acc),
        'final_val_accuracy': float(final_val_acc),
        'epochs_trained': epochs_trained,
        'params': model.count_params(),
        'model_path': model_path,
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
        }
    }
    
    # Save run metrics
    with open(os.path.join(run_dir, f'metrics_{timestamp}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n>>> {mode.upper()}-CHANNEL Results:")
    print(f"    Best Val Accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    print(f"    Epochs trained: {epochs_trained}/{config.EPOCHS}")
    
    # Don't clear session yet - we need model for per-SNR eval
    return model, history, metrics, run_dir


# =============================================================================
# MAIN COMPARISON
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
        # Write everything else to log (including newlines)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main():
    master_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up clean log file
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = f'{log_dir}/phase2_{master_timestamp}.log'
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    print("=" * 80)
    print("PHASE 2: SINGLE vs MULTI-CHANNEL COMPARISON")
    print(f"Timestamp: {master_timestamp}")
    print(f"Clean log: {log_path}")
    print("=" * 80)
    
    config = ExperimentConfig()
    config.print_config()
    
    # Master results directory (timestamped)
    results_dir = f'results_local/phase2_comparison/{master_timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save experiment config
    config_dict = {
        'master_timestamp': master_timestamp,
        'subset_fraction': config.SUBSET_FRACTION,
        'train_val_split': config.TRAIN_VAL_SPLIT,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'base_learning_rate': config.BASE_LEARNING_RATE,
        'min_learning_rate': config.MIN_LEARNING_RATE,
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'random_seed': config.RANDOM_SEED,
        'accuracy_threshold': config.ACCURACY_THRESHOLD,
        'target_snrs': TARGET_SNRS,
        'hardware_params': {
            'grid_size': GRID_SIZE,
            'gain': GAIN,
            'shifts': list(DEFAULT_SHIFTS),
        }
    }
    with open(os.path.join(results_dir, f'experiment_config_{master_timestamp}.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    results = {}
    
    # =========================================================================
    # EXPERIMENT 1: SINGLE-CHANNEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SINGLE-CHANNEL")
    print("=" * 80)
    
    single_data_dir = 'data/processed_hw_single'
    
    # Generate dataset with stratified SNR sampling
    single_counts, single_snr_dist = generate_dataset_to_disk(
        config.DATA_PATH, 
        single_data_dir, 
        mode='single', 
        subset_fraction=config.SUBSET_FRACTION,
        train_val_split=config.TRAIN_VAL_SPLIT,
        seed=config.RANDOM_SEED
    )
    
    # Train model
    model_single, history_single, metrics_single, run_dir_single = train_model(
        data_dir=single_data_dir,
        results_dir=results_dir,
        mode='single',
        seed=config.RANDOM_SEED,
        config=config
    )
    
    # Per-SNR evaluation
    print("\nEvaluating per-SNR accuracy (single-channel)...")
    single_snr_accuracy = evaluate_per_snr(
        model_single, 
        os.path.join(single_data_dir, 'validation'),
        MODEL_CLASS_NAMES
    )
    
    print("\n  Per-SNR Accuracy (Single-Channel):")
    for snr in TARGET_SNRS:
        print(f"    SNR {snr:>3}dB: {single_snr_accuracy[snr]*100:6.2f}%")
    
    results['single_channel'] = {
        'overall_accuracy': metrics_single['best_val_accuracy'],
        'snr_accuracy': {str(k): v for k, v in single_snr_accuracy.items()},
        'epochs_trained': metrics_single['epochs_trained'],
        'model_path': metrics_single['model_path'],
    }
    
    # Free memory
    del model_single
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    
    # =========================================================================
    # EXPERIMENT 2: MULTI-CHANNEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MULTI-CHANNEL")
    print("=" * 80)
    
    multi_data_dir = 'data/processed_hw_multi'
    
    # Generate dataset with stratified SNR sampling
    multi_counts, multi_snr_dist = generate_dataset_to_disk(
        config.DATA_PATH, 
        multi_data_dir, 
        mode='multi', 
        subset_fraction=config.SUBSET_FRACTION,
        train_val_split=config.TRAIN_VAL_SPLIT,
        seed=config.RANDOM_SEED
    )
    
    # Train model
    model_multi, history_multi, metrics_multi, run_dir_multi = train_model(
        data_dir=multi_data_dir,
        results_dir=results_dir,
        mode='multi',
        seed=config.RANDOM_SEED,
        config=config
    )
    
    # Per-SNR evaluation
    print("\nEvaluating per-SNR accuracy (multi-channel)...")
    multi_snr_accuracy = evaluate_per_snr(
        model_multi, 
        os.path.join(multi_data_dir, 'validation'),
        MODEL_CLASS_NAMES
    )
    
    print("\n  Per-SNR Accuracy (Multi-Channel):")
    for snr in TARGET_SNRS:
        print(f"    SNR {snr:>3}dB: {multi_snr_accuracy[snr]*100:6.2f}%")
    
    results['multi_channel'] = {
        'overall_accuracy': metrics_multi['best_val_accuracy'],
        'snr_accuracy': {str(k): v for k, v in multi_snr_accuracy.items()},
        'epochs_trained': metrics_multi['epochs_trained'],
        'model_path': metrics_multi['model_path'],
    }
    
    # Free memory
    del model_multi
    tf.keras.backend.clear_session()
    gc.collect()
    
    # =========================================================================
    # COMPARISON & DECISION
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    single_acc = results['single_channel']['overall_accuracy']
    multi_acc = results['multi_channel']['overall_accuracy']
    acc_diff = multi_acc - single_acc
    
    print(f"\n{'Overall Accuracy:':<30}")
    print(f"  Single-Channel: {single_acc*100:.2f}%")
    print(f"  Multi-Channel:  {multi_acc*100:.2f}%")
    print(f"  Difference:     {acc_diff*100:+.2f}%")
    
    print(f"\n{'Per-SNR Accuracy Comparison:':<30}")
    print(f"  {'SNR':<8} {'Single':>10} {'Multi':>10} {'Diff':>10}")
    print("  " + "-" * 40)
    for snr in TARGET_SNRS:
        s_acc = single_snr_accuracy[snr]
        m_acc = multi_snr_accuracy[snr]
        diff = m_acc - s_acc
        print(f"  {snr:>3}dB    {s_acc*100:>9.2f}% {m_acc*100:>9.2f}% {diff*100:>+9.2f}%")
    
    # Average per-SNR difference
    avg_snr_diff = np.mean([multi_snr_accuracy[snr] - single_snr_accuracy[snr] for snr in TARGET_SNRS])
    print("  " + "-" * 40)
    print(f"  {'Avg':>3}      {np.mean(list(single_snr_accuracy.values()))*100:>9.2f}% {np.mean(list(multi_snr_accuracy.values()))*100:>9.2f}% {avg_snr_diff*100:>+9.2f}%")
    
    print("\nFPGA Resource Comparison:")
    print(f"  Image Gen Compute: 1x (single) vs 3x (multi)")
    print(f"  Memory Bandwidth:  1x (single) vs 3x (multi)")
    print(f"  Kernel Storage:    1 kernel vs 3 kernels")
    
    # =========================================================================
    # FINAL RECOMMENDATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    threshold = config.ACCURACY_THRESHOLD
    
    if acc_diff <= threshold:
        recommendation = "SINGLE-CHANNEL"
        reason = f"Multi-channel advantage ({acc_diff*100:.2f}%) ≤ {threshold*100:.0f}% threshold"
    else:
        recommendation = "MULTI-CHANNEL"
        reason = f"Multi-channel advantage ({acc_diff*100:.2f}%) > {threshold*100:.0f}% threshold"
    
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
        print("    ✓ 1/3 image generation compute")
        print("    ✓ 1/3 memory bandwidth")
        print("    ✓ Simpler pipeline")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results['master_timestamp'] = master_timestamp
    results['config'] = config_dict
    
    final_results_path = os.path.join(results_dir, f'final_results_{master_timestamp}.json')
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}/")
    
    # Quick reference
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Single-Channel: {single_acc*100:.2f}%")
    print(f"Multi-Channel:  {multi_acc*100:.2f}%")
    print(f"Difference:     {acc_diff*100:+.2f}%")
    print(f"Decision:       {recommendation}")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    main()
