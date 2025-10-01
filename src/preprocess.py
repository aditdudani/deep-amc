
import h5py
import numpy as np
import tensorflow as tf
import os
import shutil
from tqdm import tqdm

# --- 1. Configuration: Define parameters for replication ---
# These parameters are derived directly from the Peng et al. (2018) paper.
HDF5_PATH = 'data/RML2018.01A_sample.h5'  # Update this path if needed
OUTPUT_DIR = 'data/processed'
import json
# Path to the fixed class order file (JSON only)
CLASSES_FIXED_JSON = 'data/classes-fixed.json'
IMAGE_SIZE = 224
SAMPLES_PER_IMAGE = 1024  # Using the full 1024 samples from the dataset frame


# Define the 8 modulation classes for replication
TARGET_MODS = [
    'BPSK', '4ASK', 'QPSK', 'OQPSK', '8PSK', '16QAM', '32QAM', '64QAM'
]
# The paper specifies a 0-10 dB range. The dataset uses 2 dB steps.
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# Define train/val split ratio (use 90% for training, 10% for validation)
TRAIN_VAL_SPLIT_RATIO = 0.9

# --- 2. Image Generation Logic (Adapted from image_generator.py) ---
def _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alpha, plane_range=7.0):
    coords = tf.linspace(-plane_range / 2.0, plane_range / 2.0, grid_size)
    grid_x, grid_y = tf.meshgrid(coords, coords)
    pixel_centers = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=1)
    iq_samples_b = iq_samples[:, tf.newaxis, :]
    pixel_centers_b = pixel_centers[tf.newaxis, :, :]
    dist_sq = tf.reduce_sum(tf.square(iq_samples_b - pixel_centers_b), axis=2)
    distances = tf.sqrt(dist_sq)
    influences = tf.exp(-alpha * distances)
    pixel_intensities = tf.reduce_sum(influences, axis=0)
    image = tf.reshape(pixel_intensities, (grid_size, grid_size))
    image_max = tf.reduce_max(image)
    if image_max > 0:
        image = image / image_max
    return image

def tf_generate_three_channel_image(iq_samples, grid_size=224, alphas=(10.0, 1.0, 0.1), plane_range=7.0):
    image_ch1 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[0], plane_range)
    image_ch2 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[1], plane_range)
    image_ch3 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[2], plane_range)
    three_channel_image = tf.stack([image_ch1, image_ch2, image_ch3], axis=-1)
    return three_channel_image

# --- 3. Main Pre-processing Logic ---
def main():

    print("--- Starting Offline Pre-processing for Peng et al. (2018) Replication ---")

    # Optionally clear output directory for a clean run
    if os.path.exists(OUTPUT_DIR):
        confirm = input(f"Output directory '{OUTPUT_DIR}' exists. Clear it before proceeding? [y/N]: ").strip().lower()
        if confirm == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print(f"Cleared '{OUTPUT_DIR}'.")
        else:
            print(f"Proceeding without clearing '{OUTPUT_DIR}'. Existing files may be overwritten or mixed.")

    # Load metadata (labels and SNRs) without loading the large 'X' dataset
    print(f"Step 1: Loading metadata from {HDF5_PATH}...")

    with h5py.File(HDF5_PATH, 'r') as hf:
        all_labels_onehot = hf['Y'][:]
        all_snrs_2d = hf['Z'][:]
        all_snrs = all_snrs_2d.flatten()  # Ensure SNRs are 1D for comparison
        # Try to get class order from HDF5, else fallback to CLASSES_FIXED_JSON
        if 'mods' in hf:
            all_mods = [mod.decode('utf-8') if isinstance(mod, bytes) else mod for mod in hf['mods'][:]]
        else:
            if not os.path.exists(CLASSES_FIXED_JSON):
                raise FileNotFoundError(f"Could not find 'mods' in HDF5 or '{CLASSES_FIXED_JSON}' for class order.")
            with open(CLASSES_FIXED_JSON, 'r') as f:
                all_mods = json.load(f)


    all_labels = np.argmax(all_labels_onehot, axis=1)
    # Map label indices to modulation names
    mod_map_from_index = {i: mod for i, mod in enumerate(all_mods)}

    # Debug prints for mapping and SNRs
    print("First 10 label indices:", all_labels[:10])
    print("First 10 SNRs:", all_snrs[:10])
    print("Class mapping (index to name):", {i: mod for i, mod in enumerate(all_mods)})
    print("Unique SNRs in file:", np.unique(all_snrs))
    print("Unique mods in first 1000 samples:", set([mod_map_from_index[l] for l in all_labels[:1000]]))

    print("Step 2: Filtering indices based on target modulations and SNRs...")
    filtered_indices_by_class = {mod: [] for mod in TARGET_MODS}


    for idx, (label_idx, snr) in enumerate(zip(all_labels, all_snrs)):
        mod = mod_map_from_index[label_idx]
        if idx < 20:
            print(f"Sample {idx}: mod={mod}, snr={snr}")
        if mod in TARGET_MODS and snr in TARGET_SNRS:
            filtered_indices_by_class[mod].append(idx)

    print("Filtering complete. Index counts per class:")
    for mod, indices in filtered_indices_by_class.items():
        print(f"  {mod}: {len(indices)}")



    print("\nStep 3: Splitting data and generating images...")
    np.random.seed(42)
    image_counts = {'train': {}, 'validation': {}}
    with h5py.File(HDF5_PATH, 'r') as hf:
        X = hf['X']
        for mod, indices in tqdm(filtered_indices_by_class.items(), desc="Classes"):
            if len(indices) == 0:
                print(f"WARNING: No samples found for class '{mod}'. Skipping.")
                continue
            indices = np.array(indices)
            np.random.shuffle(indices)
            split_point = int(len(indices) * TRAIN_VAL_SPLIT_RATIO)
            train_idx = indices[:split_point]
            val_idx = indices[split_point:]

            # Create output directories
            train_dir = os.path.join(OUTPUT_DIR, 'train', mod)
            val_dir = os.path.join(OUTPUT_DIR, 'validation', mod)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Generate training images
            image_counts['train'][mod] = 0
            for i, idx in enumerate(tqdm(train_idx, desc=f"train/{mod}", leave=False)):
                iq = X[idx][:SAMPLES_PER_IMAGE]
                iq = np.asarray(iq, dtype=np.float32)
                img = tf_generate_three_channel_image(iq, grid_size=IMAGE_SIZE)
                img = tf.clip_by_value(img, 0, 1)
                img_np = (img.numpy() * 255).astype(np.uint8)
                fname = f"{i+1:06d}.png"
                tf.keras.utils.save_img(os.path.join(train_dir, fname), img_np)
                image_counts['train'][mod] += 1

            # Generate validation images
            image_counts['validation'][mod] = 0
            for i, idx in enumerate(tqdm(val_idx, desc=f"validation/{mod}", leave=False)):
                iq = X[idx][:SAMPLES_PER_IMAGE]
                iq = np.asarray(iq, dtype=np.float32)
                img = tf_generate_three_channel_image(iq, grid_size=IMAGE_SIZE)
                img = tf.clip_by_value(img, 0, 1)
                img_np = (img.numpy() * 255).astype(np.uint8)
                fname = f"{i+1:04d}.png"
                tf.keras.utils.save_img(os.path.join(val_dir, fname), img_np)
                image_counts['validation'][mod] += 1


    print("\n--- Offline pre-processing finished successfully! ---")
    print(f"Generated images are saved in: {OUTPUT_DIR}")

    # Print summary of images generated
    print("\nSummary of images generated:")
    for split in image_counts:
        print(f"  {split}:")
        for mod, count in image_counts[split].items():
            print(f"    {mod}: {count}")

if __name__ == '__main__':
    main()
