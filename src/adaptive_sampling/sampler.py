import os
import sys
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# Ensure src/ path for script execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import IMAGE_SIZE, TARGET_SNRS


def load_metadata_csv(path: str) -> List[Tuple[str, int, int]]:
    """Load metadata CSV returning a list of (file_path, class_id, snr)."""
    rows: List[Tuple[str, int, int]] = []
    with tf.io.gfile.GFile(path, 'r') as f:
        header = f.readline().strip().split(',')
        # Expect: file_path, class_name, class_id, snr, h5_index
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            file_path = parts[0]
            class_id = int(parts[2])
            snr = int(parts[3])
            rows.append((file_path, class_id, snr))
    return rows


def build_buckets(records: List[Tuple[str, int, int]], snrs: List[int]) -> Tuple[Dict[Tuple[int, int], List[str]], Dict[int, int]]:
    """Group file paths into buckets keyed by (class_id, snr). Returns (buckets, snr_to_idx)."""
    snr_to_idx = {s: i for i, s in enumerate(snrs)}
    buckets: Dict[Tuple[int, int], List[str]] = {}
    for file_path, class_id, snr in records:
        if snr not in snr_to_idx:
            continue
        key = (class_id, snr)
        buckets.setdefault(key, []).append(file_path)
    return buckets, snr_to_idx


def init_uniform_weights(num_classes: int, snrs: List[int]) -> np.ndarray:
    w = np.ones((num_classes, len(snrs)), dtype=np.float32)
    w /= w.sum()
    return w


class WeightedSamplerSequence(tf.keras.utils.Sequence):
    def __init__(self,
                 train_metadata_csv: str,
                 class_count: int,
                 snrs: List[int] = None,
                 batch_size: int = 64,
                 epoch_size: int = None,
                 weights: np.ndarray = None,
                 shuffle_within_bucket: bool = True,
                 seed: int = 1234):
        """
        - epoch_size: number of samples per epoch; defaults to number of records in metadata.
        - weights: 2D array [num_classes, num_snrs] (will be normalized in-place).
        """
        self.records = load_metadata_csv(train_metadata_csv)
        self.num_classes = class_count
        self.snrs = snrs if snrs is not None else TARGET_SNRS
        self.batch_size = batch_size
        self.epoch_size = epoch_size or len(self.records)
        self.steps = math.ceil(self.epoch_size / self.batch_size)
        self.buckets, self.snr_to_idx = build_buckets(self.records, self.snrs)
        self.rng = random.Random(seed)
        self.shuffle_within_bucket = shuffle_within_bucket

        self.weights = weights if weights is not None else init_uniform_weights(self.num_classes, self.snrs)
        self._normalize_weights()
        self._prepare_bucket_iters()

    def _normalize_weights(self):
        total = float(self.weights.sum())
        if total <= 0:
            self.weights = init_uniform_weights(self.num_classes, self.snrs)
        else:
            self.weights = self.weights.astype(np.float32) / total

    def _prepare_bucket_iters(self):
        # Optionally shuffle file lists to avoid reading the same items.
        if self.shuffle_within_bucket:
            for key in self.buckets:
                self.rng.shuffle(self.buckets[key])

    def __len__(self):
        return self.steps

    def _draw_bucket(self) -> Tuple[int, int]:
        # Flatten weights to pick (class_id, snr_idx)
        flat = self.weights.flatten()
        choice = np.random.choice(len(flat), p=flat)
        class_id = choice // len(self.snrs)
        snr_idx = choice % len(self.snrs)
        snr = self.snrs[snr_idx]
        return class_id, snr

    def _sample_from_bucket(self, class_id: int, snr: int) -> Tuple[np.ndarray, int]:
        key = (class_id, snr)
        paths = self.buckets.get(key, [])
        if not paths:
            # Fallback: any path from the same class (ignore SNR) or any record
            class_paths = []
            for (c, s), lst in self.buckets.items():
                if c == class_id and lst:
                    class_paths.extend(lst)
            if class_paths:
                p = class_paths[self.rng.randrange(len(class_paths))]
            else:
                # absolute fallback
                p = self.records[self.rng.randrange(len(self.records))][0]
        else:
            p = paths[self.rng.randrange(len(paths))]

        # Load image: returns float32 array [H,W,3] scaled to [0,1]
        img = tf.keras.utils.load_img(p, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        arr = tf.keras.utils.img_to_array(img)  # [H,W,3], float32 in [0,255]
        arr = arr / 255.0
        return arr, class_id

    def __getitem__(self, idx):
        bs = self.batch_size
        X = np.zeros((bs, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        y = np.zeros((bs,), dtype=np.int64)
        for i in range(bs):
            c, snr = self._draw_bucket()
            arr, label = self._sample_from_bucket(c, snr)
            X[i] = arr
            y[i] = label
        return X, y

    def on_epoch_end(self):
        # No-op except reshuffle buckets if needed; weights may be updated externally
        self._prepare_bucket_iters()
