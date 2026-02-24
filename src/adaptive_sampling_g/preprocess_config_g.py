import os
import sys
import json
import shutil
import argparse
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Ensure src/ is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE

HDF5_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
OUTPUT_DIR = 'data/processed_g'
CLASSES_FIXED_JSON = 'data/classes-fixed.json'
SAMPLES_PER_IMAGE = 1024
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42
GAIN = 128


class TeeLogger:
    """Write to both stdout (with progress bars) and a clean log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        if '\r' in message and '\n' not in message:
            return
        if any(pattern in message for pattern in ['[====', 'ETA:', '━', 'it/s]']):
            return
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def parse_args():
    p = argparse.ArgumentParser(description='Generate Config G image dataset to data/processed_g')
    p.add_argument('--no-log', action='store_true', help='Disable clean log file')
    p.add_argument('--force-clear', action='store_true', help='Clear output directory without interactive prompt')
    return p.parse_args()


def create_center_weighted_cross_kernel(size=3):
    kernel = np.zeros((size, size), dtype=np.int32)
    center = size // 2
    kernel[center, :] = GAIN
    kernel[:, center] = GAIN
    kernel[center, center] = GAIN * 2
    return kernel


def hw_gen_layer(iq_samples, kernel, shift_val=0, grid_size=224):
    scale = grid_size / 7.0
    u = (iq_samples[:, 0] + 3.5) * scale
    v = (iq_samples[:, 1] + 3.5) * scale

    u_idx = np.clip(np.round(u), 0, grid_size - 1).astype(np.int16)
    v_idx = np.clip(np.round(v), 0, grid_size - 1).astype(np.int16)

    accumulator = np.zeros((grid_size, grid_size), dtype=np.int32)

    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    for x, y in zip(u_idx, v_idx):
        x_min, x_max = max(0, x - pad_h), min(grid_size, x + pad_h + 1)
        y_min, y_max = max(0, y - pad_w), min(grid_size, y + pad_w + 1)

        k_x_min = pad_h - (x - x_min)
        k_x_max = k_x_min + (x_max - x_min)
        k_y_min = pad_w - (y - y_min)
        k_y_max = k_y_min + (y_max - y_min)

        accumulator[x_min:x_max, y_min:y_max] += kernel[k_x_min:k_x_max, k_y_min:k_y_max]

    accumulator = np.clip(accumulator, -32768, 32767)
    output = accumulator >> shift_val
    return np.clip(output, 0, 255).astype(np.uint8)


def generate_config_g_image(iq_samples, grid_size=224):
    kernel = create_center_weighted_cross_kernel(3)
    ch = hw_gen_layer(iq_samples, kernel, shift_val=0, grid_size=grid_size)
    return np.stack([ch, ch, ch], axis=-1)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = None
    if not args.no_log:
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/preprocess_config_g_{timestamp}.log'
        logger = TeeLogger(log_path)
        sys.stdout = logger
        print(f"Clean log: {log_path}")

    print('--- Starting Offline Pre-processing (Config G: 3x3 Cross Centered) ---')
    try:
        if os.path.exists(OUTPUT_DIR):
            if args.force_clear:
                shutil.rmtree(OUTPUT_DIR)
                print(f"Cleared '{OUTPUT_DIR}' (force-clear).")
            else:
                confirm = input(f"Output directory '{OUTPUT_DIR}' exists. Clear it? [y/N]: ").strip().lower()
                if confirm == 'y':
                    shutil.rmtree(OUTPUT_DIR)
                    print(f"Cleared '{OUTPUT_DIR}'.")
                else:
                    print('Proceeding without clearing; files may be mixed or overwritten.')

        print(f'Loading metadata from {HDF5_PATH}...')
        with h5py.File(HDF5_PATH, 'r') as hf:
            all_labels_onehot = hf['Y'][:]
            all_snrs_2d = hf['Z'][:]
            all_snrs = all_snrs_2d.flatten()
            if 'mods' in hf:
                all_mods = [mod.decode('utf-8') if isinstance(mod, bytes) else mod for mod in hf['mods'][:]]
            else:
                if not os.path.exists(CLASSES_FIXED_JSON):
                    raise FileNotFoundError("Could not find 'mods' in HDF5 or classes-fixed.json")
                with open(CLASSES_FIXED_JSON, 'r') as f:
                    all_mods = json.load(f)

        all_labels = np.argmax(all_labels_onehot, axis=1)
        mod_map_from_index = {i: mod for i, mod in enumerate(all_mods)}
        filtered_indices_by_class = {mod: [] for mod in TARGET_MODS}

        for idx, (label_idx, snr) in enumerate(zip(all_labels, all_snrs)):
            mod = mod_map_from_index[label_idx]
            if mod in TARGET_MODS and snr in TARGET_SNRS:
                filtered_indices_by_class[mod].append(idx)

        print('Index counts per class:')
        for mod, indices in filtered_indices_by_class.items():
            print(f'  {mod}: {len(indices)}')

        np.random.seed(RANDOM_SEED)
        image_counts = {'train': {}, 'validation': {}}

        with h5py.File(HDF5_PATH, 'r') as hf:
            X = hf['X']
            for mod, indices in tqdm(filtered_indices_by_class.items(), desc='Classes'):
                if len(indices) == 0:
                    print(f"WARNING: No samples for class '{mod}'. Skipping.")
                    continue

                indices = np.array(indices)
                np.random.shuffle(indices)
                split_point = int(len(indices) * TRAIN_VAL_SPLIT_RATIO)
                train_idx = indices[:split_point]
                val_idx = indices[split_point:]

                train_dir = os.path.join(OUTPUT_DIR, 'train', mod)
                val_dir = os.path.join(OUTPUT_DIR, 'validation', mod)
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)

                image_counts['train'][mod] = 0
                for i, idx in enumerate(tqdm(train_idx, desc=f'train/{mod}', leave=False)):
                    iq = np.asarray(X[idx][:SAMPLES_PER_IMAGE], dtype=np.float32)
                    img_np = generate_config_g_image(iq, grid_size=IMAGE_SIZE)
                    fname = f"{i+1:06d}.png"
                    tf.keras.utils.save_img(os.path.join(train_dir, fname), img_np)
                    image_counts['train'][mod] += 1

                image_counts['validation'][mod] = 0
                for i, idx in enumerate(tqdm(val_idx, desc=f'validation/{mod}', leave=False)):
                    iq = np.asarray(X[idx][:SAMPLES_PER_IMAGE], dtype=np.float32)
                    img_np = generate_config_g_image(iq, grid_size=IMAGE_SIZE)
                    fname = f"{i+1:04d}.png"
                    tf.keras.utils.save_img(os.path.join(val_dir, fname), img_np)
                    image_counts['validation'][mod] += 1

        print('\nSummary of images generated (Config G):')
        for split in image_counts:
            print(f'  {split}:')
            for mod, count in image_counts[split].items():
                print(f'    {mod}: {count}')
    finally:
        if logger:
            sys.stdout = logger.terminal
            logger.close()


if __name__ == '__main__':
    main()
