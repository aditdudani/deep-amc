"""
Hardware Image Generation Calibration Script

This script finds optimal parameters for hardware-native image generation:
1. Runs shift sweeps to find values that give good dynamic range
2. Tests single vs multi-channel approaches
3. Generates sample images for visual inspection

Run this BEFORE generating the full training dataset.
"""

import os
import sys
import numpy as np
import h5py
from datetime import datetime
import json

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- HARDWARE PARAMETERS ---
GRID_SIZE = 224
GAIN = 32

def create_exp_kernel(size, alpha):
    """Create exponential decay kernel: exp(-alpha * r) * GAIN"""
    center = (size - 1) / 2.0
    y, x = np.ogrid[-center:center+1, -center:center+1]
    r = np.sqrt(x*x + y*y)
    kernel = np.exp(-alpha * r)
    kernel = (kernel / kernel.max() * GAIN).astype(np.int16)
    return kernel

# Kernels with proper alphas (alpha=10 was too aggressive - single pixel!)
# Sharp: 11x11, alpha=1.0 (tight but has spread)
# Medium: 21x21, alpha=0.3 (medium spread)
# Blur: 31x31, alpha=0.1 (wide spread)
KERNEL_SHARP = create_exp_kernel(11, alpha=1.0)
KERNEL_MEDIUM = create_exp_kernel(21, alpha=0.3)
KERNEL_BLUR = create_exp_kernel(31, alpha=0.1)

# Define kernels for each configuration
KERNELS = {
    'single': {
        'ch1': KERNEL_SHARP,
    },
    'multi': {
        'ch1': KERNEL_SHARP,
        'ch2': KERNEL_MEDIUM,
        'ch3': KERNEL_BLUR,
    }
}

# --- LABEL MAPPING (HDF5 -> Model) ---
HDF5_TO_MODEL_MAP = {
    1: 2,   # 4ASK
    3: 5,   # BPSK
    4: 7,   # QPSK
    5: 4,   # 8PSK
    12: 0,  # 16QAM
    13: 1,  # 32QAM
    14: 3,  # 64QAM
    23: 6   # OQPSK
}

MODEL_CLASS_NAMES = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
HDF5_CLASS_NAMES = {
    1: '4ASK', 3: 'BPSK', 4: 'QPSK', 5: '8PSK',
    12: '16QAM', 13: '32QAM', 14: '64QAM', 23: 'OQPSK'
}


def hardware_gen_layer(iq_samples, kernel, shift_val, grid_size=224):
    """
    Hardware-accurate image generation layer.
    Uses FIXED bit-shift scaling (no dynamic max).
    """
    # 1. Coordinate mapping: [-3.5, 3.5] -> [0, 224]
    scale = grid_size / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale
    
    # Quantize to integer indices
    u_idx = np.clip(np.round(u), 0, grid_size-1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, grid_size-1).astype(np.int16)
    
    # 2. Accumulator (int32 to avoid overflow)
    accumulator = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # 3. Kernel stamping
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    for x, y in zip(u_idx, v_idx):
        x_min = max(0, x - pad_h)
        x_max = min(grid_size, x + pad_h + 1)
        y_min = max(0, y - pad_w)
        y_max = min(grid_size, y + pad_w + 1)
        
        k_x_min = pad_h - (x - x_min)
        k_x_max = k_x_min + (x_max - x_min)
        k_y_min = pad_w - (y - y_min)
        k_y_max = k_y_min + (y_max - y_min)
        
        accumulator[x_min:x_max, y_min:y_max] += kernel[k_x_min:k_x_max, k_y_min:k_y_max]
    
    # 4. Fixed bit-shift scaling (NO dynamic max!)
    output = accumulator >> shift_val
    return np.clip(output, 0, 255).astype(np.uint8)


def compute_channel_stats(img):
    """Compute statistics for a single channel image."""
    return {
        'min': int(img.min()),
        'max': int(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std()),
        'pct_saturated': float((img == 255).mean() * 100),
        'pct_black': float((img == 0).mean() * 100),
    }


def calibration_sweep(iq_samples, kernel, shift_range=range(0, 16)):
    """
    Sweep shift values to find optimal dynamic range.
    
    Target:
    - Mean: 30-80 (good mid-range brightness)
    - Max: 200-255 (uses full dynamic range)
    - % Saturated: < 5% (not clipping signal)
    - % Black: 60-90% (background should be dark, constellation is sparse)
    """
    results = []
    for shift in shift_range:
        img = hardware_gen_layer(iq_samples, kernel, shift)
        stats = compute_channel_stats(img)
        stats['shift'] = shift
        results.append(stats)
    return results


def load_calibration_samples(data_path, n_per_class=20, snr_min=10):
    """Load samples for calibration - stratified by class."""
    print(f"Loading calibration samples from {data_path}...")
    
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:]  # (N, 2, 1024)
        Y = f['Y'][:]  # (N, 24) one-hot
        Z = f['Z'][:].flatten()  # (N,) SNR values - flatten in case 2D
    
    # Get integer labels
    y_int = np.argmax(Y, axis=1)
    
    samples = {}
    for hdf5_idx, class_name in HDF5_CLASS_NAMES.items():
        # Filter by class and SNR
        mask = (y_int == hdf5_idx) & (Z >= snr_min)
        indices = np.where(mask)[0]
        
        if len(indices) >= n_per_class:
            selected = np.random.choice(indices, n_per_class, replace=False)
        else:
            selected = indices
        
        # Convert to (N, 1024, 2) format
        iq_data = X[selected].transpose(0, 2, 1)  # (N, 1024, 2)
        samples[class_name] = iq_data
        print(f"  {class_name}: {len(selected)} samples")
    
    return samples


def run_calibration_sweep(samples, kernel_config='multi'):
    """Run calibration sweep across all classes."""
    kernels = KERNELS[kernel_config]
    
    results = {}
    for ch_name, kernel in kernels.items():
        print(f"\n=== Calibrating {ch_name} (kernel size: {kernel.shape}) ===")
        results[ch_name] = {}
        
        for class_name, iq_batch in samples.items():
            class_results = []
            for iq_samples in iq_batch:
                sweep = calibration_sweep(iq_samples, kernel, shift_range=range(0, 12))
                class_results.append(sweep)
            
            # Average across samples
            avg_results = []
            for shift_idx in range(12):
                avg = {
                    'shift': shift_idx,
                    'mean': np.mean([r[shift_idx]['mean'] for r in class_results]),
                    'max': np.mean([r[shift_idx]['max'] for r in class_results]),
                    'std': np.mean([r[shift_idx]['std'] for r in class_results]),
                    'pct_saturated': np.mean([r[shift_idx]['pct_saturated'] for r in class_results]),
                    'pct_black': np.mean([r[shift_idx]['pct_black'] for r in class_results]),
                }
                avg_results.append(avg)
            
            results[ch_name][class_name] = avg_results
    
    return results


def find_optimal_shift(sweep_results, target_mean_range=(30, 80)):
    """Find shift value that gives optimal dynamic range."""
    for result in sweep_results:
        mean = result['mean']
        pct_sat = result['pct_saturated']
        if target_mean_range[0] <= mean <= target_mean_range[1] and pct_sat < 5:
            return result['shift']
    
    # Fallback: find closest to target mean
    target = (target_mean_range[0] + target_mean_range[1]) / 2
    best = min(sweep_results, key=lambda r: abs(r['mean'] - target))
    return best['shift']


def print_sweep_table(sweep_results):
    """Print a formatted table of sweep results."""
    print(f"{'Shift':>6} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>7} {'%Sat':>6} {'%Black':>7}")
    print("-" * 50)
    for r in sweep_results:
        print(f"{r['shift']:>6} {r.get('min', 0):>5} {int(r['max']):>5} {r['mean']:>7.1f} {r['std']:>7.1f} {r['pct_saturated']:>6.1f} {r['pct_black']:>7.1f}")


def generate_sample_images(samples, shifts, kernel_config='multi', output_dir='results_local/calibration'):
    """Generate sample images with chosen shifts for visual inspection."""
    import matplotlib.pyplot as plt
    
    kernels = KERNELS[kernel_config]
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name, iq_batch in samples.items():
        # Take first 3 samples
        for idx, iq_samples in enumerate(iq_batch[:3]):
            if kernel_config == 'single':
                # Single channel - grayscale
                img = hardware_gen_layer(iq_samples, kernels['ch1'], shifts['ch1'])
                
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{class_name} - Single Channel (shift={shifts["ch1"]})')
                ax.axis('off')
            else:
                # Multi-channel - RGB
                ch1 = hardware_gen_layer(iq_samples, kernels['ch1'], shifts['ch1'])
                ch2 = hardware_gen_layer(iq_samples, kernels['ch2'], shifts['ch2'])
                ch3 = hardware_gen_layer(iq_samples, kernels['ch3'], shifts['ch3'])
                img = np.stack([ch1, ch2, ch3], axis=-1)
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Individual channels
                for i, (ch, name, shift) in enumerate(zip([ch1, ch2, ch3], ['Sharp', 'Medium', 'Coarse'], 
                                                          [shifts['ch1'], shifts['ch2'], shifts['ch3']])):
                    axes[i].imshow(ch, cmap='gray')
                    axes[i].set_title(f'{name} (shift={shift})\nmean={ch.mean():.1f}')
                    axes[i].axis('off')
                
                # Combined RGB
                axes[3].imshow(img)
                axes[3].set_title(f'{class_name} - Combined')
                axes[3].axis('off')
            
            filepath = os.path.join(output_dir, f'{class_name}_{idx}_{kernel_config}.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("HARDWARE IMAGE GENERATION CALIBRATION")
    print("=" * 60)
    
    # Configuration
    data_path = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
    n_per_class = 20  # Samples per class for calibration
    snr_min = 10      # Minimum SNR for calibration samples
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure the RadioML dataset is available.")
        return
    
    # Load calibration samples
    samples = load_calibration_samples(data_path, n_per_class, snr_min)
    
    # --- PHASE 1: Multi-channel calibration ---
    print("\n" + "=" * 60)
    print("PHASE 1: MULTI-CHANNEL CALIBRATION")
    print("=" * 60)
    
    multi_results = run_calibration_sweep(samples, kernel_config='multi')
    
    # Find optimal shifts for each channel
    multi_shifts = {}
    for ch_name in ['ch1', 'ch2', 'ch3']:
        print(f"\n--- {ch_name.upper()} Results (averaged across classes) ---")
        
        # Average across all classes
        all_class_avgs = []
        for class_name, sweep in multi_results[ch_name].items():
            all_class_avgs.append(sweep)
        
        # Compute grand average
        grand_avg = []
        for shift_idx in range(12):
            grand_avg.append({
                'shift': shift_idx,
                'mean': np.mean([c[shift_idx]['mean'] for c in all_class_avgs]),
                'max': np.mean([c[shift_idx]['max'] for c in all_class_avgs]),
                'std': np.mean([c[shift_idx]['std'] for c in all_class_avgs]),
                'pct_saturated': np.mean([c[shift_idx]['pct_saturated'] for c in all_class_avgs]),
                'pct_black': np.mean([c[shift_idx]['pct_black'] for c in all_class_avgs]),
            })
        
        print_sweep_table(grand_avg)
        
        # Different target ranges for different channels
        if ch_name == 'ch1':
            target_range = (10, 50)  # Sharp should be sparser
        elif ch_name == 'ch2':
            target_range = (20, 60)  # Medium moderate
        else:
            target_range = (30, 80)  # Coarse can be brighter
        
        optimal_shift = find_optimal_shift(grand_avg, target_range)
        multi_shifts[ch_name] = optimal_shift
        print(f"\n>>> OPTIMAL SHIFT for {ch_name}: {optimal_shift}")
    
    # --- PHASE 2: Single-channel calibration ---
    print("\n" + "=" * 60)
    print("PHASE 2: SINGLE-CHANNEL CALIBRATION")
    print("=" * 60)
    
    single_results = run_calibration_sweep(samples, kernel_config='single')
    
    single_shifts = {}
    for ch_name in ['ch1']:
        print(f"\n--- {ch_name.upper()} Results (averaged across classes) ---")
        
        all_class_avgs = []
        for class_name, sweep in single_results[ch_name].items():
            all_class_avgs.append(sweep)
        
        grand_avg = []
        for shift_idx in range(12):
            grand_avg.append({
                'shift': shift_idx,
                'mean': np.mean([c[shift_idx]['mean'] for c in all_class_avgs]),
                'max': np.mean([c[shift_idx]['max'] for c in all_class_avgs]),
                'std': np.mean([c[shift_idx]['std'] for c in all_class_avgs]),
                'pct_saturated': np.mean([c[shift_idx]['pct_saturated'] for c in all_class_avgs]),
                'pct_black': np.mean([c[shift_idx]['pct_black'] for c in all_class_avgs]),
            })
        
        print_sweep_table(grand_avg)
        
        optimal_shift = find_optimal_shift(grand_avg, (10, 50))
        single_shifts[ch_name] = optimal_shift
        print(f"\n>>> OPTIMAL SHIFT for {ch_name}: {optimal_shift}")
    
    # --- PHASE 3: Generate sample images ---
    print("\n" + "=" * 60)
    print("PHASE 3: GENERATING SAMPLE IMAGES")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results_local/calibration/{timestamp}'
    
    print(f"\nGenerating multi-channel samples with shifts: {multi_shifts}")
    generate_sample_images(samples, multi_shifts, 'multi', f'{output_dir}/multi')
    
    print(f"\nGenerating single-channel samples with shifts: {single_shifts}")
    generate_sample_images(samples, single_shifts, 'single', f'{output_dir}/single')
    
    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    
    summary = {
        'timestamp': timestamp,
        'multi_channel': {
            'shifts': multi_shifts,
            'kernels': {
                'ch1': f'{KERNEL_SHARP.shape[0]}x{KERNEL_SHARP.shape[0]} (α=1.0)',
                'ch2': f'{KERNEL_MEDIUM.shape[0]}x{KERNEL_MEDIUM.shape[0]} (α=0.3)',
                'ch3': f'{KERNEL_BLUR.shape[0]}x{KERNEL_BLUR.shape[0]} (α=0.1)',
            },
            'gain': GAIN,
        },
        'single_channel': {
            'shifts': single_shifts,
            'kernels': {
                'ch1': f'{KERNEL_SHARP.shape[0]}x{KERNEL_SHARP.shape[0]} (α=1.0)',
            },
            'gain': GAIN,
        },
        'samples_per_class': n_per_class,
        'snr_min': snr_min,
    }
    
    print(f"\nMulti-channel config:")
    print(f"  Shifts: ch1={multi_shifts['ch1']}, ch2={multi_shifts['ch2']}, ch3={multi_shifts['ch3']}")
    print(f"  Kernels: {KERNEL_SHARP.shape[0]}x{KERNEL_SHARP.shape[0]}, {KERNEL_MEDIUM.shape[0]}x{KERNEL_MEDIUM.shape[0]}, {KERNEL_BLUR.shape[0]}x{KERNEL_BLUR.shape[0]}")
    print(f"  Gain: {GAIN}")
    
    print(f"\nSingle-channel config:")
    print(f"  Shift: {single_shifts['ch1']}")
    print(f"  Kernel: {KERNEL_SHARP.shape[0]}x{KERNEL_SHARP.shape[0]}")
    print(f"  Gain: {GAIN}")
    
    # Save summary
    summary_path = f'{output_dir}/calibration_summary.json'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nSample images saved to: {output_dir}/")
    print("\n>>> NEXT STEP: Visually inspect images and run dataset generation <<<")


if __name__ == '__main__':
    main()
