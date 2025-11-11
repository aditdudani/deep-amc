import os
import sys
import json
from typing import List, Dict

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.image_generator import tf_generate_three_channel_image
from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE

HDF5_PATH = os.path.join('data', 'GOLD_XYZ_OSC.0001_1024.hdf5')
MODEL_PATH = os.path.join('models', 'squeezenet_v11_rmsprop.h5')
TRAIN_DIR = os.path.join('data', 'processed', 'train')
RESULTS_DIR = 'results'
OUT_JSON = os.path.join(RESULTS_DIR, 'accuracy_by_snr_squeezenet.json')
OUT_PNG = os.path.join(RESULTS_DIR, 'accuracy_by_snr_squeezenet.png')

ALPHAS = (10.0, 1.0, 0.1)
SAMPLES_PER_IMAGE = 1024
MAX_SAMPLES_PER_CLASS_PER_SNR = None
CHUNK_SIZE = 128
PREDICT_BATCH = 64

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

def _indices_by_mod_and_snr(h5_path: str, target_mods: List[str], target_snrs: List[int]) -> Dict[int, Dict[str, List[int]]]:
    labels, snrs, mods = _load_label_and_snr_metadata(h5_path)
    if mods is None:
        fixed_json = os.path.join('data', 'classes-fixed.json')
        if not os.path.exists(fixed_json):
            raise FileNotFoundError("Could not determine class order: 'mods' missing and classes-fixed.json not found")
        with open(fixed_json, 'r') as f:
            mods = json.load(f)
    idx_to_mod = {i: m for i, m in enumerate(mods)}
    buckets: Dict[int, Dict[str, List[int]]] = {snr: {m: [] for m in target_mods} for snr in target_snrs}
    for i, (lab, snr) in enumerate(zip(labels, snrs)):
        m = idx_to_mod[lab]
        if m in target_mods and snr in buckets:
            buckets[snr][m].append(i)
    return buckets

def _gen_images_for_indices(h5_path: str, indices: List[int]) -> np.ndarray:
    imgs = []
    with h5py.File(h5_path, 'r') as hf:
        X_dset = hf['X']
        for idx in indices:
            iq = np.asarray(X_dset[idx][:SAMPLES_PER_IMAGE], dtype=np.float32)
            img = tf_generate_three_channel_image(iq, grid_size=IMAGE_SIZE, alphas=ALPHAS)
            img = tf.clip_by_value(img, 0, 1)
            img_np = (img.numpy() * 255.0).astype(np.float32)
            imgs.append(img_np)
    return np.stack(imgs, axis=0)

def main():
    print("\n--- Evaluating SqueezeNet accuracy by SNR (common) ---\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    train_classes = _infer_class_order(TRAIN_DIR)
    print(f"Training class order ({len(train_classes)}): {train_classes}")
    mod_to_class_idx = {m: i for i, m in enumerate(train_classes)}
    buckets = _indices_by_mod_and_snr(HDF5_PATH, TARGET_MODS, TARGET_SNRS)
    acc_by_snr = {}
    for snr in TARGET_SNRS:
        total = 0
        correct = 0
        for mod, idxs in buckets[snr].items():
            if MAX_SAMPLES_PER_CLASS_PER_SNR is not None and len(idxs) > MAX_SAMPLES_PER_CLASS_PER_SNR:
                idxs = idxs[:MAX_SAMPLES_PER_CLASS_PER_SNR]
            if not idxs:
                continue
            batch = 32
            for start in range(0, len(idxs), batch):
                chunk = idxs[start:start + batch]
                X_chunk = _gen_images_for_indices(HDF5_PATH, chunk).astype(np.float32)
                y_chunk = np.full((X_chunk.shape[0],), mod_to_class_idx[mod], dtype=np.int64)
                probs = model.predict(X_chunk, batch_size=PREDICT_BATCH, verbose=0)
                y_pred = np.argmax(probs, axis=1)
                correct += int((y_pred == y_chunk).sum())
                total += y_chunk.size
        if total == 0:
            print(f"No data for SNR={snr}; skipping.")
            continue
        acc = float(correct / total)
        acc_by_snr[snr] = acc
        print(f"SNR {snr:>2} dB -> accuracy: {acc:.4f} (n={total})")
    with open(OUT_JSON, 'w') as f:
        json.dump({"accuracy_by_snr": acc_by_snr, "snrs": TARGET_SNRS, "classes": train_classes}, f, indent=2)
    print(f"Saved JSON: {OUT_JSON}")
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
