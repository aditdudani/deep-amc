
import h5py
import numpy as np
import tensorflow as tf
import os
import shutil
from tqdm import tqdm

# --- 1. Configuration: Define parameters for replication ---
# These parameters are derived directly from the Peng et al. (2018) paper.
HDF5_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'  # Update this path if needed
OUTPUT_DIR = 'data/processed'
# Path to the fixed class order file
CLASSES_FIXED_PATH = 'data/classes-fixed.txt'
IMAGE_SIZE = 224
SAMPLES_PER_IMAGE = 1024  # Using the full 1024 samples from the dataset frame


# Define the 8 modulation classes for replication
TARGET_MODS = [
    'BPSK', '4ASK', 'QPSK', 'OQPSK', '8PSK', '16QAM', '32QAM', '64QAM'
]
# The paper specifies a 0-10 dB range. The dataset uses 2 dB steps.
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# Define dataset sizes as per the paper
SAMPLES_PER_CLASS_TRAIN = 100000
SAMPLES_PER_CLASS_VAL = 1000

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
        all_snrs = hf['Z'][:]
        # Try to get class order from HDF5, else fallback to CLASSES_FIXED_PATH
        if 'mods' in hf:
            all_mods = [mod.decode('utf-8') if isinstance(mod, bytes) else mod for mod in hf['mods'][:]]
        else:
            if not os.path.exists(CLASSES_FIXED_PATH):
                raise FileNotFoundError(f"Could not find 'mods' in HDF5 or '{CLASSES_FIXED_PATH}' for class order.")
            with open(CLASSES_FIXED_PATH, 'r') as f:
                all_mods = [line.strip() for line in f if line.strip()]

    all_labels = np.argmax(all_labels_onehot, axis=1)
    # Map label indices to modulation names
    mod_map_from_index = {i: mod for i, mod in enumerate(all_mods)}

    print("Step 2: Filtering indices based on target modulations and SNRs...")
    filtered_indices_by_class = {mod: [] for mod in TARGET_MODS}

    for idx, (label_idx, snr) in enumerate(zip(all_labels, all_snrs)):
        mod = mod_map_from_index[label_idx]
        if mod in TARGET_MODS and snr in TARGET_SNRS:
            filtered_indices_by_class[mod].append(idx)

    print("Filtering complete. Index counts per class:")
    for mod, indices in filtered_indices_by_class.items():
        print(f"  {mod}: {len(indices)}")


    print("\nStep 3: Sub-sampling indices and generating images...")
    np.random.seed(42)
    train_indices = {}
    val_indices = {}
    for mod, indices in filtered_indices_by_class.items():
        if len(indices) < SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_VAL:
            print(f"WARNING: Not enough samples for class '{mod}'. Required: {SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_VAL}, Found: {len(indices)}. Skipping this class.")
            continue
        try:
            train_idx = np.random.choice(indices, SAMPLES_PER_CLASS_TRAIN, replace=False)
            val_pool = list(set(indices) - set(train_idx))
            if len(val_pool) < SAMPLES_PER_CLASS_VAL:
                print(f"WARNING: Not enough validation samples for class '{mod}'. Required: {SAMPLES_PER_CLASS_VAL}, Found: {len(val_pool)}. Skipping validation for this class.")
                continue
            val_idx = np.random.choice(val_pool, SAMPLES_PER_CLASS_VAL, replace=False)
            train_indices[mod] = train_idx
            val_indices[mod] = val_idx
        except ValueError as e:
            print(f"WARNING: Sampling error for class '{mod}': {e}. Skipping this class.")
            continue

    # Create output directories
    for split, indices_dict in [('train', train_indices), ('validation', val_indices)]:
        for mod in indices_dict:
            out_dir = os.path.join(OUTPUT_DIR, split, mod)
            os.makedirs(out_dir, exist_ok=True)

    # Track image counts for summary
    image_counts = {split: {mod: 0 for mod in indices_dict} for split, indices_dict in [('train', train_indices), ('validation', val_indices)]}

    # Open the HDF5 file once to efficiently access 'X'
    with h5py.File(HDF5_PATH, 'r') as hf:
        X = hf['X']
        for split, indices_dict, samples_per_class in [
            ('train', train_indices, SAMPLES_PER_CLASS_TRAIN),
            ('validation', val_indices, SAMPLES_PER_CLASS_VAL)
        ]:
            for mod in indices_dict:
                out_dir = os.path.join(OUTPUT_DIR, split, mod)
                indices = indices_dict[mod]
                for i, idx in enumerate(tqdm(indices, desc=f"{split}/{mod}")):
                    iq = X[idx][:SAMPLES_PER_IMAGE]  # shape: (1024, 2)
                    iq = np.asarray(iq, dtype=np.float32)
                    img = tf_generate_three_channel_image(iq, grid_size=IMAGE_SIZE)
                    img = tf.clip_by_value(img, 0, 1)
                    img_np = (img.numpy() * 255).astype(np.uint8)
                    fname = f"{i+1:06d}.png" if split == 'train' else f"{i+1:04d}.png"
                    tf.keras.utils.save_img(os.path.join(out_dir, fname), img_np)
                    image_counts[split][mod] += 1


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
