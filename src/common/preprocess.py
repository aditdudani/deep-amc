import h5py
import numpy as np
import tensorflow as tf
import os
import sys
import shutil
from tqdm import tqdm
import json

# Ensure src/ is on path so common.* imports work when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.image_generator import tf_generate_three_channel_image
from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE

HDF5_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
OUTPUT_DIR = 'data/processed'
CLASSES_FIXED_JSON = 'data/classes-fixed.json'
SAMPLES_PER_IMAGE = 1024
TRAIN_VAL_SPLIT_RATIO = 0.9

def main():
    print("--- Starting Offline Pre-processing (common) ---")
    if os.path.exists(OUTPUT_DIR):
        confirm = input(f"Output directory '{OUTPUT_DIR}' exists. Clear it? [y/N]: ").strip().lower()
        if confirm == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print(f"Cleared '{OUTPUT_DIR}'.")
        else:
            print("Proceeding without clearing; files may be mixed or overwritten.")

    print(f"Loading metadata from {HDF5_PATH}...")
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

    print("Index counts per class:")
    for mod, indices in filtered_indices_by_class.items():
        print(f"  {mod}: {len(indices)}")

    np.random.seed(42)
    image_counts = {'train': {}, 'validation': {}}
    with h5py.File(HDF5_PATH, 'r') as hf:
        X = hf['X']
        for mod, indices in tqdm(filtered_indices_by_class.items(), desc="Classes"):
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
            for i, idx in enumerate(tqdm(train_idx, desc=f"train/{mod}", leave=False)):
                iq = np.asarray(X[idx][:SAMPLES_PER_IMAGE], dtype=np.float32)
                img = tf_generate_three_channel_image(iq, grid_size=IMAGE_SIZE)
                img = tf.clip_by_value(img, 0, 1)
                img_np = (img.numpy() * 255).astype(np.uint8)
                fname = f"{i+1:06d}.png"
                tf.keras.utils.save_img(os.path.join(train_dir, fname), img_np)
                image_counts['train'][mod] += 1
            image_counts['validation'][mod] = 0
            for i, idx in enumerate(tqdm(val_idx, desc=f"validation/{mod}", leave=False)):
                iq = np.asarray(X[idx][:SAMPLES_PER_IMAGE], dtype=np.float32)
                img = tf_generate_three_channel_image(iq, grid_size=IMAGE_SIZE)
                img = tf.clip_by_value(img, 0, 1)
                img_np = (img.numpy() * 255).astype(np.uint8)
                fname = f"{i+1:04d}.png"
                tf.keras.utils.save_img(os.path.join(val_dir, fname), img_np)
                image_counts['validation'][mod] += 1

    print("\nSummary of images generated:")
    for split in image_counts:
        print(f"  {split}:")
        for mod, count in image_counts[split].items():
            print(f"    {mod}: {count}")

if __name__ == '__main__':
    main()
