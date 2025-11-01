import os
import json
from typing import List, Dict

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from image_generator import tf_generate_three_channel_image


# --------- Config (no CLI) ---------
HDF5_PATH = os.path.join('data', 'GOLD_XYZ_OSC.0001_1024.hdf5')
MODEL_PATH = os.path.join('models', 'squeezenet_v11_rmsprop.h5')
TRAIN_DIR = os.path.join('data', 'processed', 'train')  # used to infer class order
RESULTS_DIR = 'results'
OUT_JSON = os.path.join(RESULTS_DIR, 'accuracy_by_snr_squeezenet.json')
OUT_PNG = os.path.join(RESULTS_DIR, 'accuracy_by_snr_squeezenet.png')

# Scope (matches your current 8-class subset)
TARGET_MODS = ['BPSK', '4ASK', 'QPSK', 'OQPSK', '8PSK', '16QAM', '32QAM', '64QAM']
TARGET_SNRS = [0, 2, 4, 6, 8, 10]
IMAGE_SIZE = 224
ALPHAS = (10.0, 1.0, 0.1)
SAMPLES_PER_IMAGE = 1024
MAX_SAMPLES_PER_CLASS_PER_SNR = None  # e.g., set to 200 for a quicker sweep


def _infer_class_order(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()
    return classes


def _load_label_and_snr_metadata(h5_path: str):
    with h5py.File(h5_path, 'r') as hf:
        Y_onehot = hf['Y'][:]
        Z_2d = hf['Z'][:]
        mods = None
        if 'mods' in hf:
            mods = [m.decode('utf-8') if isinstance(m, bytes) else m for m in hf['mods'][:]]
        labels = np.argmax(Y_onehot, axis=1)
        snrs = Z_2d.flatten()
    return labels, snrs, mods


def _indices_by_mod_and_snr(h5_path: str,
                             target_mods: List[str],
                             target_snrs: List[int]) -> Dict[int, Dict[str, List[int]]]:
    labels, snrs, mods = _load_label_and_snr_metadata(h5_path)

    # If HDF5 doesn't include class names, fall back to fixed order json if present
    if mods is None:
        fixed_json = os.path.join('data', 'classes-fixed.json')
        if not os.path.exists(fixed_json):
            raise FileNotFoundError("Could not determine class order: 'mods' missing in HDF5 and data/classes-fixed.json not found")
        with open(fixed_json, 'r') as f:
            mods = json.load(f)

    idx_to_mod = {i: m for i, m in enumerate(mods)}
    buckets: Dict[int, Dict[str, List[int]]] = {snr: {m: [] for m in target_mods} for snr in target_snrs}
    for i, (lab, snr) in enumerate(zip(labels, snrs)):
        m = idx_to_mod[lab]
        if m in target_mods and snr in buckets:
            buckets[snr][m].append(i)
    return buckets


def _gen_images_for_indices(h5_path: str, indices: List[int], samples_per_image: int, image_size: int, alphas) -> np.ndarray:
    imgs = []
    with h5py.File(h5_path, 'r') as hf:
        X_dset = hf['X']
        for idx in indices:
            iq = np.asarray(X_dset[idx][:samples_per_image], dtype=np.float32)
            img = tf_generate_three_channel_image(iq, grid_size=image_size, alphas=alphas)
            img = tf.clip_by_value(img, 0, 1)
            img_np = (img.numpy() * 255.0).astype(np.float32)  # model expects 0..255; Rescaling layer handles 1/255
            imgs.append(img_np)
    return np.stack(imgs, axis=0)


def main():
    print("\n--- Evaluating SqueezeNet accuracy by SNR ---\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Match class order to training directories
    train_classes = _infer_class_order(TRAIN_DIR)
    print(f"Training class order ({len(train_classes)}): {train_classes}")
    mod_to_class_idx = {m: i for i, m in enumerate(train_classes)}

    # Bucket indices per SNR and modulation
    buckets = _indices_by_mod_and_snr(HDF5_PATH, TARGET_MODS, TARGET_SNRS)

    acc_by_snr: Dict[int, float] = {}

    for snr in TARGET_SNRS:
        all_imgs = []
        all_labels = []
        for mod, idxs in buckets[snr].items():
            if MAX_SAMPLES_PER_CLASS_PER_SNR is not None and len(idxs) > MAX_SAMPLES_PER_CLASS_PER_SNR:
                idxs = idxs[:MAX_SAMPLES_PER_CLASS_PER_SNR]
            if not idxs:
                continue
            imgs = _gen_images_for_indices(HDF5_PATH, idxs, SAMPLES_PER_IMAGE, IMAGE_SIZE, ALPHAS)
            labels = np.full((imgs.shape[0],), mod_to_class_idx[mod], dtype=np.int64)
            all_imgs.append(imgs)
            all_labels.append(labels)

        if not all_imgs:
            print(f"No data for SNR={snr}; skipping.")
            continue

        X = np.concatenate(all_imgs, axis=0).astype(np.float32)
        y_true = np.concatenate(all_labels, axis=0)

        probs = model.predict(X, batch_size=64, verbose=0)  # Softmax head
        y_pred = np.argmax(probs, axis=1)
        acc = float((y_pred == y_true).mean())
        acc_by_snr[snr] = acc
        print(f"SNR {snr:>2} dB -> accuracy: {acc:.4f} (n={len(y_true)})")

    # Persist results
    with open(OUT_JSON, 'w') as f:
        json.dump({"accuracy_by_snr": acc_by_snr, "snrs": TARGET_SNRS, "classes": train_classes}, f, indent=2)
    print(f"Saved JSON: {OUT_JSON}")

    # Plot
    snrs = sorted(acc_by_snr.keys())
    accs = [acc_by_snr[s] for s in snrs]
    plt.figure(figsize=(7, 4))
    plt.plot(snrs, accs, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('SqueezeNet: Accuracy vs SNR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"Saved plot: {OUT_PNG}")


if __name__ == '__main__':
    main()
