"""
Phase 1: Rigorous Parameter Calibration for Hardware Image Generation

This script runs a systematic sweep to find optimal shift values that:
1. Target Mean: 30-80 (good mid-range brightness)
2. Target Max: 200-255 (uses full dynamic range)
3. % Saturated: < 5% (not clipping signal info)
4. % Black: 60-90% (background should be dark - constellation is sparse)

Run this BEFORE generating the full dataset to lock in the shift parameters.
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.data_loader import load_data_sample

# =============================================================================
# HARDWARE PARAMETERS (MUST MATCH simulate_hardware.py)
# =============================================================================
GRID_SIZE = 224
GAIN = 128

# Kernels - same as simulate_hardware.py
KERNEL_SHARP = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.int16) * GAIN

KERNEL_MEDIUM = np.array([[1, 2, 1],
                          [2, 8, 2],
                          [1, 2, 1]], dtype=np.int16) * GAIN

KERNEL_BLUR = np.ones((11, 11), dtype=np.int16) * GAIN

# All kernels for sweep
KERNELS = {
    'ch1_sharp': {'kernel': KERNEL_SHARP, 'size': '3x3'},
    'ch2_medium': {'kernel': KERNEL_MEDIUM, 'size': '3x3'},
    'ch3_blur': {'kernel': KERNEL_BLUR, 'size': '11x11'},
}

# Label info
VALID_HDF5_CLASSES = [1, 3, 4, 5, 12, 13, 14, 23]
HDF5_CLASS_NAMES = {
    1: '4ASK', 3: 'BPSK', 4: 'QPSK', 5: '8PSK',
    12: '16QAM', 13: '32QAM', 14: '64QAM', 23: 'OQPSK'
}

# =============================================================================
# TARGET CRITERIA
# =============================================================================
TARGET_MEAN_MIN = 30
TARGET_MEAN_MAX = 80
TARGET_MAX_MIN = 200
TARGET_MAX_MAX = 255
TARGET_SATURATED_MAX = 5.0  # percent
TARGET_BLACK_MIN = 60.0     # percent
TARGET_BLACK_MAX = 90.0     # percent

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

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


def compute_stats(img):
    """Compute all relevant statistics for a single-channel image."""
    return {
        'min': int(img.min()),
        'max': int(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std()),
        'pct_saturated': float((img == 255).mean() * 100),
        'pct_black': float((img == 0).mean() * 100),
    }


def calibration_sweep(iq_samples, kernel, shift_range=range(0, 12)):
    """
    Run calibration sweep for a single kernel across multiple samples.
    Returns aggregated statistics per shift value.
    """
    results = defaultdict(list)
    
    for shift in shift_range:
        for iq in iq_samples:
            img = hardware_gen_layer(iq, kernel, shift)
            stats = compute_stats(img)
            results[shift].append(stats)
    
    # Aggregate across samples
    aggregated = {}
    for shift, stats_list in results.items():
        aggregated[shift] = {
            'min': np.mean([s['min'] for s in stats_list]),
            'max': np.mean([s['max'] for s in stats_list]),
            'mean': np.mean([s['mean'] for s in stats_list]),
            'std': np.mean([s['std'] for s in stats_list]),
            'pct_saturated': np.mean([s['pct_saturated'] for s in stats_list]),
            'pct_black': np.mean([s['pct_black'] for s in stats_list]),
        }
    return aggregated


def score_shift(stats):
    """
    Score a shift configuration. Lower is better.
    Returns score and whether it passes all criteria.
    """
    score = 0
    passes = True
    
    # Mean in target range
    if stats['mean'] < TARGET_MEAN_MIN:
        score += (TARGET_MEAN_MIN - stats['mean']) ** 2
        passes = False
    elif stats['mean'] > TARGET_MEAN_MAX:
        score += (stats['mean'] - TARGET_MEAN_MAX) ** 2
        passes = False
    
    # Max should be high (near 255)
    if stats['max'] < TARGET_MAX_MIN:
        score += (TARGET_MAX_MIN - stats['max']) ** 2
        passes = False
    
    # Saturation should be low
    if stats['pct_saturated'] > TARGET_SATURATED_MAX:
        score += (stats['pct_saturated'] - TARGET_SATURATED_MAX) ** 2
        passes = False
    
    # Black percentage in range
    if stats['pct_black'] < TARGET_BLACK_MIN:
        score += (TARGET_BLACK_MIN - stats['pct_black']) ** 2
        passes = False
    elif stats['pct_black'] > TARGET_BLACK_MAX:
        score += (stats['pct_black'] - TARGET_BLACK_MAX) ** 2
        passes = False
    
    return score, passes


def find_optimal_shift(sweep_results):
    """Find the best shift value from sweep results."""
    best_shift = None
    best_score = float('inf')
    any_passes = False
    
    for shift, stats in sweep_results.items():
        score, passes = score_shift(stats)
        if passes and not any_passes:
            any_passes = True
            best_shift = shift
            best_score = score
        elif passes and score < best_score:
            best_shift = shift
            best_score = score
        elif not any_passes and score < best_score:
            best_shift = shift
            best_score = score
    
    return best_shift, any_passes


def print_sweep_table(sweep_results, channel_name):
    """Print formatted sweep results table."""
    print(f"\n{'='*80}")
    print(f"{channel_name} SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Shift':>6} {'Min':>6} {'Max':>6} {'Mean':>8} {'Std':>8} {'%Sat':>7} {'%Black':>8} {'Pass':>6}")
    print("-" * 80)
    
    for shift in sorted(sweep_results.keys()):
        s = sweep_results[shift]
        _, passes = score_shift(s)
        pass_str = "YES" if passes else "no"
        print(f"{shift:>6} {s['min']:>6.1f} {s['max']:>6.1f} {s['mean']:>8.2f} {s['std']:>8.2f} "
              f"{s['pct_saturated']:>7.2f} {s['pct_black']:>8.2f} {pass_str:>6}")


def run_per_class_calibration(X, Y, Z, kernel, channel_name, n_per_class=20):
    """Run calibration sweep separately for each modulation class."""
    print(f"\n{'#'*80}")
    print(f"PER-CLASS CALIBRATION: {channel_name}")
    print(f"{'#'*80}")
    
    y_int = np.argmax(Y, axis=1) if Y.ndim > 1 else Y.flatten()
    
    class_results = {}
    for class_idx in VALID_HDF5_CLASSES:
        class_name = HDF5_CLASS_NAMES[class_idx]
        
        # Get samples for this class (high SNR only)
        mask = (y_int == class_idx) & (Z[:, 0] > 10)
        class_indices = np.where(mask)[0]
        
        if len(class_indices) < n_per_class:
            print(f"  WARNING: {class_name} has only {len(class_indices)} samples")
            continue
        
        np.random.shuffle(class_indices)
        samples = X[class_indices[:n_per_class]]
        
        # Run sweep
        sweep = calibration_sweep(samples, kernel, shift_range=range(0, 10))
        optimal, passes = find_optimal_shift(sweep)
        
        class_results[class_name] = {
            'optimal_shift': optimal,
            'passes_criteria': passes,
            'stats_at_optimal': sweep[optimal],
        }
        
        print(f"\n  {class_name}: Optimal Shift = {optimal} (Passes: {passes})")
        print(f"    Mean={sweep[optimal]['mean']:.1f}, Max={sweep[optimal]['max']:.1f}, "
              f"%Sat={sweep[optimal]['pct_saturated']:.1f}, %Black={sweep[optimal]['pct_black']:.1f}")
    
    # Check consistency
    shifts = [r['optimal_shift'] for r in class_results.values()]
    if len(set(shifts)) == 1:
        print(f"\n  ✓ All classes agree on shift = {shifts[0]}")
    else:
        print(f"\n  ⚠ Classes disagree on shift: {shifts}")
        print(f"    Using median: {int(np.median(shifts))}")
    
    return class_results


def generate_sample_images(X, y_int, shifts, save_dir, n_per_class=3):
    """Generate and save sample images with the chosen shifts."""
    os.makedirs(save_dir, exist_ok=True)
    
    for class_idx in VALID_HDF5_CLASSES:
        class_name = HDF5_CLASS_NAMES[class_idx]
        mask = (y_int == class_idx)
        class_indices = np.where(mask)[0]
        
        if len(class_indices) == 0:
            continue
        
        np.random.shuffle(class_indices)
        
        for i in range(min(n_per_class, len(class_indices))):
            iq = X[class_indices[i]]
            
            ch1 = hardware_gen_layer(iq, KERNEL_SHARP, shifts['ch1'])
            ch2 = hardware_gen_layer(iq, KERNEL_MEDIUM, shifts['ch2'])
            ch3 = hardware_gen_layer(iq, KERNEL_BLUR, shifts['ch3'])
            img = np.stack([ch1, ch2, ch3], axis=-1)
            
            filename = f'{save_dir}/{class_name}_{i}.png'
            plt.imsave(filename, img)
            print(f"  Saved: {filename}")


# =============================================================================
# MAIN CALIBRATION ROUTINE
# =============================================================================

def main():
    print("=" * 80)
    print("HARDWARE IMAGE GENERATION - PHASE 1: PARAMETER CALIBRATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  GRID_SIZE: {GRID_SIZE}")
    print(f"  GAIN: {GAIN}")
    print(f"  Kernels: Sharp(3x3), Medium(3x3), Blur(11x11)")
    print(f"\nTarget Criteria:")
    print(f"  Mean: {TARGET_MEAN_MIN}-{TARGET_MEAN_MAX}")
    print(f"  Max: {TARGET_MAX_MIN}-{TARGET_MAX_MAX}")
    print(f"  %Saturated: < {TARGET_SATURATED_MAX}%")
    print(f"  %Black: {TARGET_BLACK_MIN}-{TARGET_BLACK_MAX}%")
    
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    try:
        X, Y, Z = load_data_sample(data_path)
        y_int = np.argmax(Y, axis=1) if Y.ndim > 1 else Y.flatten()
        print(f"  Loaded {len(X)} samples")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    # Get diverse calibration samples (all classes, high SNR)
    mask = np.isin(y_int, VALID_HDF5_CLASSES) & (Z[:, 0] > 10)
    valid_idx = np.where(mask)[0]
    np.random.shuffle(valid_idx)
    
    n_cal = min(200, len(valid_idx))  # 200 samples for global sweep
    cal_samples = X[valid_idx[:n_cal]]
    print(f"  Using {n_cal} calibration samples (SNR > 10)")
    
    # =========================================================================
    # GLOBAL SWEEP (averaged across all modulations)
    # =========================================================================
    print("\n" + "=" * 80)
    print("GLOBAL SWEEP (Averaged Across All Modulations)")
    print("=" * 80)
    
    optimal_shifts = {}
    
    for ch_name, ch_info in KERNELS.items():
        print(f"\nProcessing {ch_name} ({ch_info['size']})...")
        sweep = calibration_sweep(cal_samples, ch_info['kernel'], shift_range=range(0, 12))
        print_sweep_table(sweep, ch_name.upper())
        
        optimal, passes = find_optimal_shift(sweep)
        optimal_shifts[ch_name] = {
            'shift': optimal,
            'passes': passes,
            'stats': sweep[optimal]
        }
        
        print(f"\n>>> OPTIMAL SHIFT for {ch_name}: {optimal} (Passes criteria: {passes})")
    
    # =========================================================================
    # PER-CLASS ANALYSIS (check consistency across modulations)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-CLASS ANALYSIS (Checking Consistency Across Modulations)")
    print("=" * 80)
    
    per_class = {}
    for ch_name, ch_info in KERNELS.items():
        per_class[ch_name] = run_per_class_calibration(
            X, Y, Z, ch_info['kernel'], ch_name.upper(), n_per_class=20
        )
    
    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDED CONFIGURATION")
    print("=" * 80)
    
    final_shifts = {
        'ch1': optimal_shifts['ch1_sharp']['shift'],
        'ch2': optimal_shifts['ch2_medium']['shift'],
        'ch3': optimal_shifts['ch3_blur']['shift'],
    }
    
    print(f"\n  GAIN = {GAIN}")
    print(f"  Ch1 (Sharp 3x3):   shift = {final_shifts['ch1']}")
    print(f"  Ch2 (Medium 3x3):  shift = {final_shifts['ch2']}")
    print(f"  Ch3 (Blur 11x11):  shift = {final_shifts['ch3']}")
    print(f"\n  DEFAULT_SHIFTS = ({final_shifts['ch1']}, {final_shifts['ch2']}, {final_shifts['ch3']})")
    
    # =========================================================================
    # SAVE RESULTS & SAMPLE IMAGES
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_local/calibration/{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config
    config = {
        'timestamp': timestamp,
        'gain': GAIN,
        'grid_size': GRID_SIZE,
        'recommended_shifts': final_shifts,
        'global_sweep': {k: {
            'shift': v['shift'],
            'passes': v['passes'],
            'mean': v['stats']['mean'],
            'max': v['stats']['max'],
            'pct_saturated': v['stats']['pct_saturated'],
            'pct_black': v['stats']['pct_black'],
        } for k, v in optimal_shifts.items()},
        'targets': {
            'mean': [TARGET_MEAN_MIN, TARGET_MEAN_MAX],
            'max': [TARGET_MAX_MIN, TARGET_MAX_MAX],
            'pct_saturated_max': TARGET_SATURATED_MAX,
            'pct_black': [TARGET_BLACK_MIN, TARGET_BLACK_MAX],
        }
    }
    
    config_path = f'{results_dir}/calibration_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved config to: {config_path}")
    
    # Generate sample images
    print(f"\n  Generating sample images...")
    generate_sample_images(X, y_int, final_shifts, f'{results_dir}/samples', n_per_class=3)
    
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Visually inspect images in: {results_dir}/samples/")
    print(f"  2. If images look good, update DEFAULT_SHIFTS in simulate_hardware.py")
    print(f"  3. Run Phase 2: Architecture decision (single vs multi-channel)")
    print(f"  4. Generate full training dataset")


if __name__ == "__main__":
    main()
