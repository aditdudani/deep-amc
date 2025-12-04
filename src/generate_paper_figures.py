"""
Generate publication-quality figures for paper.

Figure 1: Single class, single SNR - shows 3 alpha channels side-by-side + 3-channel composite
Figure 2: Single class across 3 different SNRs - showing effect of noise

Usage:
    python src/generate_paper_figures.py

Modify the configuration section below to change class, SNR, and output settings.
"""

import os
import sys
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.image_generator import _tf_iq_to_enhanced_gray_image, tf_generate_three_channel_image

# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# Path to the full RadioML dataset
HDF5_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
CLASSES_JSON = 'data/classes-fixed.json'

# Image generation parameters (same as your training)
ALPHAS = (10.0, 1.0, 0.1)
IMAGE_SIZE = 224
PLANE_RANGE = 7.0

# Output directory for figures
OUTPUT_DIR = 'results_local/paper_figures'

# ============= FIGURE 1 CONFIGURATION =============
# Single class, single SNR - showing alpha channel decomposition
# RECOMMENDATION: QPSK at 10dB - clear 4-point constellation, shows alpha effects well
FIG1_CLASS = 'QPSK'      # Modulation class to use
FIG1_SNR = 10            # SNR in dB (10 dB for clean constellation)
FIG1_FRAME_INDEX = 0     # Which frame to use (0-4095 available per class/SNR)

# ============= FIGURE 2 CONFIGURATION =============
# Single class across multiple SNRs - showing noise effect
# RECOMMENDATION: 8PSK - clear 8-point ring, degrades gracefully with noise
# Avoid 16QAM - becomes indistinct blob at low SNR
FIG2_CLASS = '8PSK'      # Modulation class (different from Fig 1)
FIG2_SNRS = [0, 6, 10] # Three SNR values: low/medium/high contrast
FIG2_FRAME_INDEX = 0     # Which frame to use

# =============================================================================
# END CONFIGURATION
# =============================================================================


def load_classes():
    """Load the correct class order from JSON file."""
    with open(CLASSES_JSON, 'r') as f:
        return json.load(f)


def get_sample_indices(h5_path, target_class, target_snr, classes):
    """
    Get indices of all frames matching the target class and SNR.
    
    Returns list of indices into the X dataset.
    """
    with h5py.File(h5_path, 'r') as hf:
        Y_onehot = hf['Y'][:]
        Z_snr = hf['Z'][:].flatten().astype(np.int32)  # SNR stored as float, convert to int
    
    labels = np.argmax(Y_onehot, axis=1)
    class_idx = classes.index(target_class)
    
    matching_indices = np.where((labels == class_idx) & (Z_snr == target_snr))[0]
    return matching_indices.tolist()


def verify_sample(h5_path, index, classes):
    """Verify what class and SNR a sample index actually corresponds to."""
    with h5py.File(h5_path, 'r') as hf:
        Y_onehot = hf['Y'][index]
        Z_snr = int(hf['Z'][index].flatten()[0])
    label = np.argmax(Y_onehot)
    class_name = classes[label]
    return class_name, Z_snr


def load_iq_sample(h5_path, index):
    """Load a single IQ sample from the dataset."""
    with h5py.File(h5_path, 'r') as hf:
        iq = np.asarray(hf['X'][index], dtype=np.float32)
    return iq


def generate_figure_1(h5_path, mod_class, snr, frame_idx, classes, output_dir):
    """
    Generate Figure 1: Alpha channel decomposition for a single sample.
    
    Creates a single figure with 4 images:
    - Three individual alpha channel images (grayscale)
    - The combined 3-channel RGB representation
    """
    print(f"\n--- Generating Figure 1 ---")
    print(f"Requested: Class={mod_class}, SNR={snr} dB, Frame={frame_idx}")
    
    # Get sample indices for this class/SNR
    indices = get_sample_indices(h5_path, mod_class, snr, classes)
    if len(indices) == 0:
        raise ValueError(f"No samples found for class={mod_class}, SNR={snr}")
    if frame_idx >= len(indices):
        raise ValueError(f"Frame index {frame_idx} out of range (max: {len(indices)-1})")
    
    sample_index = indices[frame_idx]
    
    # Verify the sample is correct
    actual_class, actual_snr = verify_sample(h5_path, sample_index, classes)
    print(f"Dataset index {sample_index} -> Actual: Class={actual_class}, SNR={actual_snr} dB")
    if actual_class != mod_class or actual_snr != snr:
        print(f"  WARNING: Mismatch detected!")
    
    # Load IQ samples
    iq_samples = load_iq_sample(h5_path, sample_index)
    iq_tf = tf.constant(iq_samples, dtype=tf.float32)
    
    # Generate individual channel images
    channels = []
    for alpha in ALPHAS:
        img = _tf_iq_to_enhanced_gray_image(iq_tf, IMAGE_SIZE, alpha, PLANE_RANGE)
        channels.append(img.numpy())
    
    # Generate 3-channel image
    three_channel = tf_generate_three_channel_image(iq_tf, IMAGE_SIZE, ALPHAS, PLANE_RANGE)
    three_channel_np = three_channel.numpy()
    
    # =========== Figure 1: All in one figure (4 images) ===========
    fig1, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, (ax, alpha) in enumerate(zip(axes[:3], ALPHAS)):
        ax.imshow(channels[i], cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
    
    axes[3].imshow(three_channel_np)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save PNG only
    os.makedirs(output_dir, exist_ok=True)
    path_1 = os.path.join(output_dir, f'fig1_{mod_class}_SNR{snr}.png')
    fig1.savefig(path_1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path_1}")
    plt.close(fig1)
    
    return sample_index


def generate_figure_2(h5_path, mod_class, snrs, frame_idx, classes, output_dir):
    """
    Generate Figure 2: Same class across multiple SNRs.
    
    Shows how the constellation diagram changes with noise level.
    """
    print(f"\n--- Generating Figure 2 ---")
    print(f"Requested: Class={mod_class}, SNRs={snrs}, Frame={frame_idx}")
    
    images = []
    used_indices = []
    
    for snr in snrs:
        indices = get_sample_indices(h5_path, mod_class, snr, classes)
        if len(indices) == 0:
            raise ValueError(f"No samples found for class={mod_class}, SNR={snr}")
        if frame_idx >= len(indices):
            raise ValueError(f"Frame index {frame_idx} out of range for SNR={snr}")
        
        sample_index = indices[frame_idx]
        used_indices.append(sample_index)
        
        # Verify the sample is correct
        actual_class, actual_snr = verify_sample(h5_path, sample_index, classes)
        print(f"SNR {snr} dB: index {sample_index} -> Actual: Class={actual_class}, SNR={actual_snr} dB")
        if actual_class != mod_class or actual_snr != snr:
            print(f"  WARNING: Mismatch detected!")
        
        iq_samples = load_iq_sample(h5_path, sample_index)
        iq_tf = tf.constant(iq_samples, dtype=tf.float32)
        
        three_channel = tf_generate_three_channel_image(iq_tf, IMAGE_SIZE, ALPHAS, PLANE_RANGE)
        images.append(three_channel.numpy())
    
    # Create figure with 3 images side by side
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, img, snr in zip(axes, images, snrs):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save PNG only
    os.makedirs(output_dir, exist_ok=True)
    snr_str = '_'.join([str(s) for s in snrs])
    path_2 = os.path.join(output_dir, f'fig2_{mod_class}_SNR_{snr_str}.png')
    fig2.savefig(path_2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path_2}")
    plt.close(fig2)
    
    return used_indices


def print_available_options(h5_path, classes):
    """Print available classes and SNRs in the dataset."""
    print("\n" + "="*60)
    print("AVAILABLE OPTIONS IN DATASET")
    print("="*60)
    
    print(f"\nClasses ({len(classes)} total):")
    for i, c in enumerate(classes):
        print(f"  {i:2d}: {c}")
    
    print(f"\nSNR values: -20 to +30 dB (steps of 2)")
    print(f"Available SNRs: {list(range(-20, 32, 2))}")
    print(f"\nFrames per class/SNR combination: 4096 (indices 0-4095)")
    print("="*60 + "\n")


def main():
    print("\n" + "="*60)
    print("PAPER FIGURE GENERATOR")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(HDF5_PATH):
        print(f"\nERROR: Dataset not found at {HDF5_PATH}")
        print("Please ensure you have the RadioML 2018.01A dataset.")
        return
    
    # Load class order
    classes = load_classes()
    
    # Print available options for reference
    print_available_options(HDF5_PATH, classes)
    
    # Print current configuration
    print("CURRENT CONFIGURATION:")
    print("-" * 40)
    print(f"Figure 1: {FIG1_CLASS} @ {FIG1_SNR} dB, frame {FIG1_FRAME_INDEX}")
    print(f"Figure 2: {FIG2_CLASS} @ {FIG2_SNRS} dB, frame {FIG2_FRAME_INDEX}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Alphas: {ALPHAS}")
    print("-" * 40)
    
    # Generate figures
    try:
        idx1 = generate_figure_1(HDF5_PATH, FIG1_CLASS, FIG1_SNR, FIG1_FRAME_INDEX, 
                                  classes, OUTPUT_DIR)
        idx2 = generate_figure_2(HDF5_PATH, FIG2_CLASS, FIG2_SNRS, FIG2_FRAME_INDEX, 
                                  classes, OUTPUT_DIR)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nFigure 1 generated for: {FIG1_CLASS} @ SNR={FIG1_SNR} dB")
        print(f"  Dataset index used: {idx1}")
        print(f"\nFigure 2 generated for: {FIG2_CLASS} @ SNRs={FIG2_SNRS} dB")
        print(f"  Dataset indices used: {idx2}")
        print(f"\nAll figures saved to: {OUTPUT_DIR}")
        print("\nFiles generated:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.startswith('fig'):
                print(f"  - {f}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == '__main__':
    main()
