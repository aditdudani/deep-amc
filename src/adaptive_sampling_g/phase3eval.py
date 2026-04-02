import os
import sys
import json
import argparse
from typing import List, Dict

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE

HDF5_PATH = os.path.join('data', 'GOLD_XYZ_OSC.0001_1024.hdf5')
MODEL_PATH = os.path.join('results', 'adaptive_sampling_g', '20260328_180649', 'model.h5')  # Update to latest
TRAIN_DIR = os.path.join('data', 'processed_g', 'train')
RESULTS_DIR = os.path.join('results', 'evaluations', 'phase3_baseline')

SAMPLES_PER_IMAGE = 1024
CHUNK_SIZE = 128
PREDICT_BATCH = 64
MAX_SAMPLES_PER_CLASS_PER_SNR = None
GAIN = 128

# --- HARDWARE EMULATION LOGIC (Config G - K20 Center-Weighted) ---
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

# --- EVALUATION LOGIC ---
def _infer_class_order(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()
    return classes


def _load_label_and_snr_metadata(h5_path: str):
    print(f"Loading metadata from {h5_path}...")
    with h5py.File(HDF5_PATH, 'r') as hf:
        labels_onehot = hf['Y'][:]
        labels = np.argmax(labels_onehot, axis=1)
        snrs = hf['Z'][:].flatten()
        if 'mods' in hf:
            mods = [m.decode('utf-8') if isinstance(m, bytes) else m for m in hf['mods'][:]]
        else:
            with open('data/classes-fixed.json', 'r') as f:
                mods = json.load(f)
    return labels, snrs, mods


def _indices_by_mod_and_snr(h5_path: str, target_mods: List[str], target_snrs: List[int]) -> Dict[int, Dict[str, List[int]]]:
    labels, snrs, mods = _load_label_and_snr_metadata(h5_path)
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
        X_all = hf['X']
        for idx in indices:
            iq = np.asarray(X_all[idx][:SAMPLES_PER_IMAGE], dtype=np.float32)
            imgs.append(generate_config_g_image(iq, grid_size=IMAGE_SIZE).astype(np.float32))
    return np.stack(imgs, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Config G model accuracy by SNR (impartial full-set methodology)')
    p.add_argument('--model-path', type=str, default=MODEL_PATH)
    p.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    p.add_argument('--predict-batch', type=int, default=PREDICT_BATCH)
    p.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    p.add_argument('--max-samples-per-class-per-snr', type=int, default=MAX_SAMPLES_PER_CLASS_PER_SNR)
    return p.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, 'phase3_accuracy_by_snr.json')
    out_png = os.path.join(args.results_dir, 'phase3_accuracy_by_snr.png')

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Match class order used during training
    train_classes = _infer_class_order(TRAIN_DIR)
    print(f"Training class order ({len(train_classes)}): {train_classes}")
    mod_to_class_idx = {mod: i for i, mod in enumerate(train_classes)}

    buckets = _indices_by_mod_and_snr(HDF5_PATH, TARGET_MODS, TARGET_SNRS)

    acc_by_snr = {}
    counts_by_snr = {}
    overall_correct = 0
    overall_total = 0

    print("\n--- Starting Phase 3 SNR Evaluation ---")
    for snr in sorted(TARGET_SNRS):
        correct = 0
        total = 0

        for mod in TARGET_MODS:
            idxs = buckets[snr][mod]
            if args.max_samples_per_class_per_snr is not None and len(idxs) > args.max_samples_per_class_per_snr:
                idxs = idxs[:args.max_samples_per_class_per_snr]
            if not idxs:
                continue

            for start in range(0, len(idxs), args.chunk_size):
                chunk = idxs[start:start + args.chunk_size]
                X_chunk = _gen_images_for_indices(HDF5_PATH, chunk).astype(np.float32)
                y_chunk = np.full((X_chunk.shape[0],), mod_to_class_idx[mod], dtype=np.int64)

                probs = model.predict(X_chunk, batch_size=args.predict_batch, verbose=0)
                y_pred = np.argmax(probs, axis=1)

                correct += int((y_pred == y_chunk).sum())
                total += y_chunk.size

        if total == 0:
            print(f"No data for SNR={snr}; skipping.")
            continue

        if total > 0:
            acc = float(correct / total)
            acc_by_snr[snr] = acc
            print(f"SNR {snr:>2} dB -> accuracy: {acc*100:.2f}% (n={total})")
            counts_by_snr[snr] = total
            overall_correct += correct
            overall_total += total

    overall_acc = float(overall_correct / overall_total) if overall_total else 0.0
    print(f"Overall accuracy across all evaluated target samples: {overall_acc*100:.2f}% (n={overall_total})")

    with open(out_json, 'w') as f:
        json.dump({
            "accuracy_by_snr": acc_by_snr,
            "counts_by_snr": counts_by_snr,
            "overall_accuracy": overall_acc,
            "overall_count": overall_total,
            "snrs": TARGET_SNRS,
            "classes": train_classes,
            "model_path": args.model_path,
        }, f, indent=2)

    # --- PLOTTING ---
    snrs = sorted(acc_by_snr.keys())
    accs = [acc_by_snr[s] * 100 for s in snrs]
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, accs, marker='o', linewidth=2, color='tab:red', label='Phase 3: Baseline K20')
    plt.axhline(y=overall_acc * 100, color='gray', linestyle='--', label=f'Overall ({overall_acc*100:.2f}%)')
    plt.title('Baseline Phase 3 Hardware Evaluation (Config G)', fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(snrs)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved JSON: {out_json}")
    print(f"Saved plot: {out_png}")

if __name__ == '__main__':
    main()
