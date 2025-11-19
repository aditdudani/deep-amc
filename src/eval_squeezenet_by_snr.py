import os
import json
import argparse
from datetime import datetime
from typing import List, Dict

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from image_generator import tf_generate_three_channel_image

# Defaults (can be overridden via CLI)
DEFAULT_HDF5 = os.path.join('data', 'GOLD_XYZ_OSC.0001_1024.hdf5')
DEFAULT_MODEL = os.path.join('models', 'squeezenet_v11_rmsprop.h5')
DEFAULT_TRAIN_DIR = os.path.join('data', 'processed', 'train')
DEFAULT_TARGET_MODS = ['BPSK', '4ASK', 'QPSK', 'OQPSK', '8PSK', '16QAM', '32QAM', '64QAM']
DEFAULT_TARGET_SNRS = [0, 2, 4, 6, 8, 10]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_ALPHAS = (10.0, 1.0, 0.1)
DEFAULT_SAMPLES_PER_IMAGE = 1024
DEFAULT_CHUNK_SIZE = 128
DEFAULT_PREDICT_BATCH = 64


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate SqueezeNet accuracy by SNR (with optional per-class detail)')
    p.add_argument('--hdf5', type=str, default=DEFAULT_HDF5, help='RadioML HDF5 path')
    p.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model .h5 path to evaluate')
    p.add_argument('--train-dir', type=str, default=DEFAULT_TRAIN_DIR, help='Directory to infer class order')
    p.add_argument('--out-base', type=str, default=os.path.join('results', 'evals', 'squeezenet'), help='Base output dir')
    p.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument('--samples-per-image', type=int, default=DEFAULT_SAMPLES_PER_IMAGE)
    p.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help='Streaming chunk size for image generation')
    p.add_argument('--predict-batch', type=int, default=DEFAULT_PREDICT_BATCH, help='Batch size for model.predict')
    p.add_argument('--limit-per-bucket', type=int, default=None, help='Max samples per (class,SNR); None=all')
    p.add_argument('--target-mods', type=str, default=','.join(DEFAULT_TARGET_MODS), help='Comma-separated modulation list')
    p.add_argument('--target-snrs', type=str, default=','.join(str(s) for s in DEFAULT_TARGET_SNRS), help='Comma-separated SNR list')
    p.add_argument('--per-class', action='store_true', help='Include per-(class,SNR) accuracy matrix in JSON output')
    p.add_argument('--alphas', type=str, default='10.0,1.0,0.1', help='Comma-separated alphas for image generator')
    return p.parse_args()


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
    """Generate a numpy batch of images for the provided indices.
    NOTE: For streaming usage only on small slices; use CHUNK_SIZE to keep memory bounded.
    """
    imgs = []
    with h5py.File(h5_path, 'r') as hf:
        X_dset = hf['X']
        for idx in indices:
            iq = np.asarray(X_dset[idx][:samples_per_image], dtype=np.float32)
            img = tf_generate_three_channel_image(iq, grid_size=image_size, alphas=alphas)
            img = tf.clip_by_value(img, 0, 1)
            img_np = (img.numpy() * 255.0).astype(np.float32)
            imgs.append(img_np)
    return np.stack(imgs, axis=0)


def main():
    args = parse_args()
    target_mods = [m.strip() for m in args.target_mods.split(',') if m.strip()]
    target_snrs = [int(s.strip()) for s in args.target_snrs.split(',') if s.strip()]
    alphas = tuple(float(a.strip()) for a in args.alphas.split(',') if a.strip())

    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.out_base, run_tag)
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, 'accuracy_by_snr_squeezenet.json')
    out_png = os.path.join(results_dir, 'accuracy_by_snr_squeezenet.png')

    print("\n--- Evaluating SqueezeNet accuracy by SNR ---\n")
    print(f"Model: {args.model}")
    print(f"HDF5: {args.hdf5}")
    print(f"Classes inferred from: {args.train_dir}")
    print(f"Target mods: {target_mods}")
    print(f"Target SNRs: {target_snrs}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    train_classes = _infer_class_order(args.train_dir)
    print(f"Training class order ({len(train_classes)}): {train_classes}")
    mod_to_class_idx = {m: i for i, m in enumerate(train_classes)}

    buckets = _indices_by_mod_and_snr(args.hdf5, target_mods, target_snrs)
    acc_by_snr: Dict[int, float] = {}
    per_class_matrix: Dict[int, Dict[str, float]] = {}

    limit = args.limit_per_bucket
    chunk_size = max(1, int(args.chunk_size))
    predict_batch = max(1, int(args.predict_batch))

    for snr in target_snrs:
        total = 0
        correct = 0
        per_class_correct = {m: 0 for m in target_mods}
        per_class_total = {m: 0 for m in target_mods}
        for mod, idxs in buckets[snr].items():
            if limit is not None and len(idxs) > limit:
                idxs = idxs[:limit]
            if not idxs:
                continue
            class_id = mod_to_class_idx[mod]
            for start in range(0, len(idxs), chunk_size):
                chunk = idxs[start:start + chunk_size]
                X_chunk = _gen_images_for_indices(args.hdf5, chunk, args.samples_per_image, args.image_size, alphas).astype(np.float32)
                y_chunk = np.full((X_chunk.shape[0],), class_id, dtype=np.int64)
                probs = model.predict(X_chunk, batch_size=predict_batch, verbose=0)
                y_pred = np.argmax(probs, axis=1)
                match = (y_pred == y_chunk)
                n_match = int(match.sum())
                correct += n_match
                total += y_chunk.size
                per_class_correct[mod] += n_match
                per_class_total[mod] += y_chunk.size
        if total == 0:
            print(f"No data for SNR={snr}; skipping.")
            continue
        acc = float(correct / total)
        acc_by_snr[snr] = acc
        if args.per_class:
            per_class_matrix[snr] = {m: (per_class_correct[m] / per_class_total[m]) if per_class_total[m] > 0 else None for m in target_mods}
        print(f"SNR {snr:>2} dB -> accuracy: {acc:.4f} (n={total})")

    payload = {
        'accuracy_by_snr': acc_by_snr,
        'snrs': target_snrs,
        'classes': train_classes,
        'target_mods': target_mods,
        'model_path': args.model,
        'limit_per_bucket': limit,
    }
    if args.per_class:
        payload['per_class_accuracy_by_snr'] = per_class_matrix

    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON: {out_json}")

    snrs_plotted = sorted(acc_by_snr.keys())
    accs = [acc_by_snr[s] for s in snrs_plotted]
    plt.figure(figsize=(7, 4))
    plt.plot(snrs_plotted, accs, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('SqueezeNet: Accuracy vs SNR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot: {out_png}")


if __name__ == '__main__':
    main()
