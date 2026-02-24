"""Reconstruct mapping from Config G PNG images to (class_id, class_name, snr, h5_index).

We mirror the logic in preprocess.py (seed=42, same TARGET_MODS/SNRS) without regenerating images.
Outputs:
    data/processed_g/metadata_train.csv
    data/processed_g/metadata_val.csv
    data/processed_g/metadata_summary.json
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

import h5py
import numpy as np
import tensorflow as tf

# Ensure 'src' is on sys.path so `common` package is importable when running as a file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE

HDF5_PATH = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'
OUTPUT_DIR = 'data/processed_g'
TRAIN_SUBDIR = 'train'
VAL_SUBDIR = 'validation'
TRAIN_CSV = os.path.join(OUTPUT_DIR, 'metadata_train.csv')
VAL_CSV = os.path.join(OUTPUT_DIR, 'metadata_val.csv')
SUMMARY_JSON = os.path.join(OUTPUT_DIR, 'metadata_summary.json')

TRAIN_VAL_SPLIT_RATIO = 0.9  # must match preprocess.py
RANDOM_SEED = 42


class TeeLogger:
    """Write to both stdout and a clean log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        if '\r' in message and '\n' not in message:
            return
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def parse_args():
    p = argparse.ArgumentParser(description='Build metadata CSVs for Config G dataset')
    p.add_argument('--no-log', action='store_true', help='Disable clean log file')
    return p.parse_args()


@dataclass
class Record:
    file_path: str
    class_name: str
    class_id: int
    snr: int
    h5_index: int


def _load_labels_and_snrs(h5_path: str):
    with h5py.File(h5_path, 'r') as hf:
        Y_onehot = hf['Y'][:]
        Z_2d = hf['Z'][:]
        if 'mods' in hf:
            mods = [m.decode('utf-8') if isinstance(m, bytes) else m for m in hf['mods'][:]]
        else:
            fixed_json = os.path.join('data', 'classes-fixed.json')
            if not os.path.exists(fixed_json):
                raise FileNotFoundError("Class order not found: neither 'mods' in HDF5 nor classes-fixed.json present")
            with open(fixed_json, 'r') as f:
                mods = json.load(f)
        labels = np.argmax(Y_onehot, axis=1)
        snrs = Z_2d.flatten()
    return labels, snrs, mods


def _filter_indices(labels, snrs, mods) -> Dict[str, List[int]]:
    mod_map = {i: m for i, m in enumerate(mods)}
    filtered: Dict[str, List[int]] = {m: [] for m in TARGET_MODS}
    for idx, (lab, snr) in enumerate(zip(labels, snrs)):
        m = mod_map[lab]
        if m in TARGET_MODS and snr in TARGET_SNRS:
            filtered[m].append(idx)
    return filtered


def _split_indices_per_class(filtered_indices: Dict[str, List[int]]):
    np.random.seed(RANDOM_SEED)
    splits = {}
    for mod, indices in filtered_indices.items():
        arr = np.array(indices)
        np.random.shuffle(arr)
        split_point = int(len(arr) * TRAIN_VAL_SPLIT_RATIO)
        splits[mod] = {
            'train': arr[:split_point].tolist(),
            'val': arr[split_point:].tolist(),
        }
    return splits


from typing import Tuple


def _build_records(splits: Dict[str, Dict[str, List[int]]], labels, snrs, mods) -> Tuple[List[Record], List[Record]]:
    # Align class_id with directory order used by image_dataset_from_directory
    train_dir_root = os.path.join(OUTPUT_DIR, TRAIN_SUBDIR)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir_root,
        labels='inferred',
        label_mode='int',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=64,
        shuffle=False,
    )
    dir_class_names = train_ds.class_names
    mod_to_id = {m: i for i, m in enumerate(dir_class_names)}
    train_records: List[Record] = []
    val_records: List[Record] = []
    for mod in TARGET_MODS:
        class_id = mod_to_id[mod]
        # Train
        train_dir = os.path.join(OUTPUT_DIR, TRAIN_SUBDIR, mod)
        for i, h5_index in enumerate(splits[mod]['train']):
            snr = int(snrs[h5_index])
            # Filenames in preprocess: sequential starting at 1, padded to 6 digits for train
            fname = f"{i+1:06d}.png"
            file_path = os.path.join(train_dir, fname)
            train_records.append(Record(file_path, mod, class_id, snr, int(h5_index)))
        # Val
        val_dir = os.path.join(OUTPUT_DIR, VAL_SUBDIR, mod)
        for i, h5_index in enumerate(splits[mod]['val']):
            snr = int(snrs[h5_index])
            # Validation filenames padded to 4 digits in preprocess
            fname = f"{i+1:04d}.png"
            file_path = os.path.join(val_dir, fname)
            val_records.append(Record(file_path, mod, class_id, snr, int(h5_index)))
    return train_records, val_records


def _write_csv(path: str, records: List[Record]):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_path', 'class_name', 'class_id', 'snr', 'h5_index'])
        for r in records:
            w.writerow([r.file_path, r.class_name, r.class_id, r.snr, r.h5_index])


def _write_summary(path: str, train_records: List[Record], val_records: List[Record]):
    summary = {
        'train': {},
        'validation': {},
        'target_mods': TARGET_MODS,
        'target_snrs': TARGET_SNRS,
        'image_size': IMAGE_SIZE,
    }
    for split_name, recs in [('train', train_records), ('validation', val_records)]:
        by_class_snr: Dict[str, Dict[int, int]] = {}
        for mod in TARGET_MODS:
            by_class_snr[mod] = {snr: 0 for snr in TARGET_SNRS}
        for r in recs:
            by_class_snr[r.class_name][r.snr] += 1
        summary[split_name]['per_class_snr_counts'] = by_class_snr
        summary[split_name]['total'] = len(recs)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = None
    if not args.no_log:
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/dataset_metadata_g_{timestamp}.log'
        logger = TeeLogger(log_path)
        sys.stdout = logger
        print(f"Clean log: {log_path}")

    try:
        if not os.path.exists(OUTPUT_DIR):
            raise FileNotFoundError(f"Processed images directory not found: {OUTPUT_DIR}")
        labels, snrs, mods = _load_labels_and_snrs(HDF5_PATH)
        filtered = _filter_indices(labels, snrs, mods)
        splits = _split_indices_per_class(filtered)
        train_records, val_records = _build_records(splits, labels, snrs, mods)

        _write_csv(TRAIN_CSV, train_records)
        _write_csv(VAL_CSV, val_records)
        _write_summary(SUMMARY_JSON, train_records, val_records)
        print(f"Wrote metadata CSVs:\n  {TRAIN_CSV}\n  {VAL_CSV}\nSummary: {SUMMARY_JSON}")
    finally:
        if logger:
            sys.stdout = logger.terminal
            logger.close()


if __name__ == '__main__':
    main()
