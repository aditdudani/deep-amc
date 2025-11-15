import os
import sys
import json
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

# Ensure src/ is on path for script-style execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import TARGET_SNRS


def load_validation_metadata(path: str) -> List[Tuple[str, int, int]]:
    rows = []
    with tf.io.gfile.GFile(path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            file_path = parts[0]
            class_id = int(parts[2])
            snr = int(parts[3])
            rows.append((file_path, class_id, snr))
    return rows


def _predict_in_batches(model, items: List[Tuple[str, int, int]], batch_size: int = 64):
    X_batch = []
    y_true = []
    preds = []
    for fp, class_id, _ in items:
        img = tf.keras.utils.load_img(fp, target_size=(model.input_shape[1], model.input_shape[2]))
        arr = tf.keras.utils.img_to_array(img)  # model applies Rescaling(1/255)
        X_batch.append(arr)
        y_true.append(class_id)
        if len(X_batch) == batch_size:
            logits = model.predict(np.array(X_batch, dtype=np.float32), verbose=0)
            preds.extend(np.argmax(logits, axis=1))
            X_batch, y_true = [], y_true
    if X_batch:
        logits = model.predict(np.array(X_batch, dtype=np.float32), verbose=0)
        preds.extend(np.argmax(logits, axis=1))
    return np.array(y_true, dtype=np.int64), np.array(preds, dtype=np.int64)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


class ConfusionBySNRCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 val_metadata_csv: str,
                 weights_ref: np.ndarray,
                 out_dir: str = 'results/adaptive_sampling',
                 beta: float = 0.3,
                 epsilon: float = 0.02,
                 max_cap: float = 0.4,
                 batch_size: int = 64,
                 snrs: List[int] = None,
                 warmup_epochs: int = 3):
        super().__init__()
        self.val_metadata_csv = val_metadata_csv
        self.weights_ref = weights_ref  # external mutable weights array
        self.out_dir = out_dir
        self.beta = beta
        self.epsilon = epsilon
        self.max_cap = max_cap
        self.batch_size = batch_size
        self.snrs = snrs if snrs is not None else TARGET_SNRS
        self.warmup_epochs = max(int(warmup_epochs), 0)
        os.makedirs(self.out_dir, exist_ok=True)
        self.val_items = load_validation_metadata(self.val_metadata_csv)

    def on_epoch_end(self, epoch, logs=None):
        # Optional warmup: don't alter weights for first N epochs
        if (epoch + 1) <= self.warmup_epochs:
            weights_path = os.path.join(self.out_dir, f'weights_epoch{epoch+1}.json')
            with open(weights_path, 'w') as f:
                json.dump({'epoch': epoch+1,
                           'weights': self.weights_ref.tolist(),
                           'snrs': self.snrs,
                           'note': f'warmup_no_update_until_epoch_{self.warmup_epochs}'}, f, indent=2)
            print(f"[ConfusionBySNR] Warmup epoch {epoch+1}/{self.warmup_epochs}: weights not updated. Saved {weights_path}")
            return

        # Group validation items by SNR
        by_snr: Dict[int, List[Tuple[str, int, int]]] = {s: [] for s in self.snrs}
        for fp, cid, snr in self.val_items:
            if snr in by_snr:
                by_snr[snr].append((fp, cid, snr))

        num_classes = int(self.weights_ref.shape[0])
        per_class_snr_acc = np.zeros_like(self.weights_ref)
        confusion_per_snr: Dict[int, List[List[int]]] = {}

        for snr in self.snrs:
            items = by_snr.get(snr, [])
            if not items:
                continue
            y_true, y_pred = _predict_in_batches(self.model, items, batch_size=self.batch_size)
            cm = _confusion_matrix(y_true, y_pred, num_classes)
            confusion_per_snr[snr] = cm.tolist()
            # per-class accuracy for this SNR
            class_totals = cm.sum(axis=1)
            class_correct = np.diag(cm)
            acc_class = np.divide(class_correct, np.maximum(class_totals, 1), dtype=np.float32)
            snr_idx = self.snrs.index(snr)
            per_class_snr_acc[:, snr_idx] = acc_class

        # Update weights: error = 1 - acc; smooth & cap
        errors = 1.0 - per_class_snr_acc
        updated = (1 - self.beta) * self.weights_ref + self.beta * (errors + self.epsilon)
        # Cap individual bucket weight to prevent collapse
        flat = updated.flatten()
        flat = np.minimum(flat, self.max_cap)
        # Renormalize
        flat /= flat.sum() if flat.sum() > 0 else 1.0
        self.weights_ref[:] = flat.reshape(self.weights_ref.shape)

        # Persist artifacts
        weights_path = os.path.join(self.out_dir, f'weights_epoch{epoch+1}.json')
        confusion_path = os.path.join(self.out_dir, f'confusion_epoch{epoch+1}.json')
        with open(weights_path, 'w') as f:
            json.dump({'epoch': epoch+1,
                       'weights': self.weights_ref.tolist(),
                       'snrs': self.snrs}, f, indent=2)
        with open(confusion_path, 'w') as f:
            json.dump({'epoch': epoch+1,
                       'confusion_per_snr': confusion_per_snr,
                       'snrs': self.snrs}, f, indent=2)
        print(f"[ConfusionBySNR] Updated weights saved: {weights_path}")
