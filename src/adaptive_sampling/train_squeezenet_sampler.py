import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
from typing import Optional, List

# Silence TensorFlow C++ logs by default (unless user overrides)
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO and WARNING

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

"""Train SqueezeNet with (optionally) adaptive sampler.

Modes:
- parity (default): use tf.data directory pipeline (baseline-identical)
- sampler-uniform: use WeightedSamplerSequence with uniform bucket weights
- adaptive: sampler + confusion-by-SNR weight updates (with warmup and gate)

Run examples:
    python src/adaptive_sampling/train_squeezenet_sampler.py --mode parity

    python src/adaptive_sampling/train_squeezenet_sampler.py --mode sampler-uniform

    python src/adaptive_sampling/train_squeezenet_sampler.py --mode adaptive --warmup-epochs 3 --min-val-acc 0.15
"""

# Ensure src/ is on sys.path so `common` imports work when run as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.squeezenet import build_squeezenet_v11
from common.config import IMAGE_SIZE as CFG_IMAGE_SIZE, TARGET_SNRS
from adaptive_sampling.sampler import (
    WeightedSamplerSequence,
    init_uniform_weights,
    load_metadata_csv,
    build_buckets,
)
from adaptive_sampling.callbacks_confusion_snr import ConfusionBySNRCallback
from adaptive_sampling.debug_utils import debug_train_steps, val_probe, ValProbeCallback


# --------------------
# Defaults; can be overridden via CLI
# --------------------
DATA_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
IMAGE_SIZE = CFG_IMAGE_SIZE
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-2
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join('results', 'adaptive_sampling', RUN_TAG)
MODEL_OUT = os.path.join('models', f'squeezenet_sampler_{RUN_TAG}.h5')
LOG_CSV = os.path.join(RESULTS_DIR, 'squeezenet_sampler_train_log.csv')
TB_LOGDIR = os.path.join(RESULTS_DIR, 'logs')
TRAIN_META_CSV = os.path.join(DATA_DIR, 'metadata_train.csv')
VAL_META_CSV = os.path.join(DATA_DIR, 'metadata_val.csv')


class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        current_lr = lr(self.model.optimizer.iterations) if callable(lr) else float(tf.keras.backend.get_value(lr))
        print(f"\n[Epoch {epoch+1}] Learning rate: {current_lr:.6g}")


def make_datasets(train_dir, val_dir, image_size, batch_size, class_names=None):
    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        class_names=class_names,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names,
    )

    # Prefetch for performance; SqueezeNet model handles normalization (Rescaling layer)
    train_ds = train_ds.prefetch(AUTOTUNE).with_options(options)
    val_ds = val_ds.prefetch(AUTOTUNE).with_options(options)
    return train_ds, val_ds


def parse_args():
    p = argparse.ArgumentParser(description='Train SqueezeNet with optional adaptive sampler')
    p.add_argument('--mode', type=str, default='parity', choices=['parity', 'sampler-uniform', 'adaptive'],
                   help='Training mode: baseline-parity (tf.data), sampler-uniform, or adaptive')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--lr', type=float, default=LEARNING_RATE)
    p.add_argument('--clipnorm', type=float, default=0.0, help='Optional gradient clipnorm for optimizer (0 disables)')
    p.add_argument('--image-size', type=int, default=IMAGE_SIZE)
    p.add_argument('--warmup-epochs', type=int, default=3, help='Adaptive: epochs to skip updates')
    p.add_argument('--min-val-acc', type=float, default=0.15, help='Adaptive: min val_acc to allow updates')
    p.add_argument('--metadata-train', type=str, default=TRAIN_META_CSV)
    p.add_argument('--metadata-val', type=str, default=VAL_META_CSV)
    p.add_argument('--verify-metadata', action='store_true', help='Quickly verify metadata paths/SNRs before sampler/adaptive runs')
    p.add_argument('--debug-sampler', action='store_true', help='Dump a preview batch from sampler and basic stats')
    p.add_argument('--epoch-size', type=int, default=None, help='Sampler: number of samples per epoch (for quick tests)')
    p.add_argument('--sampler-backend', type=str, default='sequence', choices=['sequence', 'tfdata', 'tfdata-dir-class'],
                   help='Use Keras Sequence or tf.data backend for sampler modes')
    p.add_argument('--resample-each-epoch', action='store_true',
                   help='For tf.data sampler backends: rebuild sampled dataset each epoch (recommended)')
    p.add_argument('--uniform-scope', type=str, default='class', choices=['class', 'class_snr'],
                   help='Uniform over classes (default) or over (class,SNR) buckets')
    p.add_argument('--snr-filter', type=str, default=None,
                   help='Comma-separated SNR list to sample from (e.g., "6,8,10"). If omitted, use all TARGET_SNRS')
    p.add_argument('--debug-trainstep', action='store_true', help='Run a few train_on_batch steps to verify learning')
    p.add_argument('--valprobe-batches', type=int, default=0, help='Probe validation preds over first N batches before/after training')
    return p.parse_args()
def _verify_metadata_quick(csv_path: str, class_names):
    """Lightweight checks: file exists, folder-based class in class_names, SNR in TARGET_SNRS.
    Scans up to ~5000 entries for speed and prints summary warnings.
    """
    import csv as _csv
    total = 0
    missing = 0
    bad_class = 0
    bad_snr = 0
    try:
        with tf.io.gfile.GFile(csv_path, 'r') as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            for i, row in enumerate(reader):
                if len(row) < 4:
                    continue
                fp = row[0]
                try:
                    snr = int(row[3])
                except Exception:
                    snr = None
                total += 1
                if not tf.io.gfile.exists(fp):
                    missing += 1
                folder = os.path.basename(os.path.dirname(fp))
                if folder not in class_names:
                    bad_class += 1
                if snr not in TARGET_SNRS:
                    bad_snr += 1
                if i >= 5000:
                    break
    except Exception as e:
        print(f"[verify-metadata] Failed to read {csv_path}: {e}")
        return
    if total == 0:
        print(f"[verify-metadata] No rows read from {csv_path}")
        return
    print(f"[verify-metadata] Checked {total} rows from {csv_path} | missing_files={missing}, bad_class_names={bad_class}, bad_snrs={bad_snr}")


def _make_tfdata_from_sampler_draws(train_meta_csv: str, class_names: list, weights: np.ndarray,
                                    epoch_size: int, batch_size: int, image_size: int,
                                    uniform_scope: str = 'class', snr_filter: Optional[List[int]] = None):
    # Build buckets consistent with WeightedSamplerSequence
    records = load_metadata_csv(train_meta_csv)
    class_name_to_id = {name: i for i, name in enumerate(class_names)}
    snrs_use = snr_filter if snr_filter is not None else TARGET_SNRS
    buckets, snr_to_idx = build_buckets(records, snrs_use, class_name_to_id)

    # Also build per-class lists (ignore SNR) for class-only uniform
    class_to_paths = {i: [] for i in range(len(class_names))}
    for (cid, _snr), lst in buckets.items():
        if lst:
            class_to_paths[cid].extend(lst)

    # Pre-sample a list of (path, label) for this epoch
    num_classes = len(class_names)
    paths = []
    labels = []
    rng = np.random.default_rng()
    if uniform_scope == 'class_snr':
        num_snrs = len(snrs_use)
        if weights.shape != (num_classes, num_snrs):
            flat = np.ones((num_classes, num_snrs), dtype=np.float32)
            flat = (flat / flat.sum()).flatten()
        else:
            flat = (weights / max(float(weights.sum()), 1e-8)).flatten()
        for _ in range(epoch_size):
            choice = rng.choice(len(flat), p=flat)
            cid = int(choice // num_snrs)
            snr_idx = int(choice % num_snrs)
            snr = snrs_use[snr_idx]
            key = (cid, snr)
            lst = buckets.get(key, [])
            if not lst:
                c_list = class_to_paths.get(cid, [])
                if c_list:
                    pth = c_list[rng.integers(0, len(c_list))]
                else:
                    all_any = sum((v for v in class_to_paths.values()), [])
                    pth = all_any[rng.integers(0, len(all_any))]
            else:
                pth = lst[rng.integers(0, len(lst))]
            paths.append(pth)
            labels.append(cid)
    else:
        # Uniform over classes; within class, keep natural SNR mix among selected snrs
        non_empty = [k for k, v in class_to_paths.items() if v]
        for _ in range(epoch_size):
            cid = int(rng.choice(non_empty))
            c_list = class_to_paths[cid]
            pth = c_list[rng.integers(0, len(c_list))]
            paths.append(pth)
            labels.append(cid)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load_fn(pth, y):
        img_bytes = tf.io.read_file(pth)
        img = tf.io.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32)
        return img, tf.cast(y, tf.int32)
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.shuffle(min(10000, epoch_size)).map(_load_fn, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds

def _make_tfdata_class_uniform_from_dir(train_dir: str, class_names: list, epoch_size: int, batch_size: int, image_size: int):
    # Build per-class file lists by scanning directory
    per_class = {i: [] for i in range(len(class_names))}
    for i, cname in enumerate(class_names):
        cdir = os.path.join(train_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for root, _dirs, files in os.walk(cdir):
            for fn in files:
                if fn.lower().endswith('.png'):
                    per_class[i].append(os.path.join(root, fn))
    non_empty = [k for k, v in per_class.items() if v]
    if not non_empty:
        raise RuntimeError("No training images found when scanning directory for class-uniform tf.data")

    # Pre-sample (path,label)
    rng = np.random.default_rng()
    paths, labels = [], []
    for _ in range(epoch_size):
        cid = int(rng.choice(non_empty))
        lst = per_class[cid]
        pth = lst[rng.integers(0, len(lst))]
        paths.append(pth)
        labels.append(cid)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load_fn(pth, y):
        img_bytes = tf.io.read_file(pth)
        img = tf.io.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32)
        return img, tf.cast(y, tf.int32)
    AUTOTUNE = tf.data.AUTOTUNE
    return ds.shuffle(min(10000, epoch_size)).map(_load_fn, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)



def main():
    args = parse_args()

    # Further reduce Python-side TF logs (keep training progress)
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass

    tf.keras.mixed_precision.set_global_policy('float32')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TB_LOGDIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Expected directories: {TRAIN_DIR} and {VAL_DIR}")

    # Determine class_names once from train dir; force same mapping for val
    image_size = int(args.image_size)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    learning_rate = float(args.lr)

    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    train_ds, val_ds = make_datasets(TRAIN_DIR, VAL_DIR, image_size, batch_size, class_names=class_names)

    # Verify val contains same set
    val_set = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])
    print(f"Train classes ({len(class_names)}): {class_names}")
    print(f"Val   classes ({len(val_set)}): {val_set}")
    if class_names != val_set:
        raise RuntimeError(
            "Train/Val class folders differ. This will break label alignment.\n"
            f"Train: {class_names}\nVal:   {val_set}")
    num_classes = len(class_names)

    # Build model
    model = build_squeezenet_v11(input_shape=(image_size, image_size, 3), num_classes=num_classes, dropout_rate=0.0)
    clipnorm = float(args.clipnorm)
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=clipnorm if clipnorm > 0 else None)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )

    ckpt = callbacks.ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    )
    csv = callbacks.CSVLogger(LOG_CSV)
    tb = callbacks.TensorBoard(log_dir=TB_LOGDIR)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    print("\n--- Training SqueezeNet v1.1 with adaptive sampler ---")

    fit_kwargs = {
        'validation_data': val_ds,
        'epochs': epochs,
        'callbacks': [ckpt, csv, tb, reduce_lr, LrPrinter()],
        'verbose': 1,
    }

    # Optional probe before training
    if args.valprobe_batches and args.mode != 'parity':
        val_probe(model, val_ds, class_names, n_batches=int(args.valprobe_batches), tag='pre')

    if args.mode == 'parity':
        model.fit(train_ds, **fit_kwargs)
    else:
        # Sampler-based modes require metadata CSVs
        if not os.path.exists(args.metadata_train) or not os.path.exists(args.metadata_val):
            raise FileNotFoundError(
                "Metadata CSVs not found. Please generate them first via:\n"
                "  PYTHONPATH=src python src/adaptive_sampling/dataset_metadata.py\n"
                f"Expected: {args.metadata_train} and {args.metadata_val}")

        if args.verify_metadata:
            _verify_metadata_quick(args.metadata_train, class_names)
            _verify_metadata_quick(args.metadata_val, class_names)

        # Initialize external weights array (mutable) for sampler and callback to share
        weights = init_uniform_weights(num_classes, TARGET_SNRS)
        if args.sampler_backend == 'sequence':
            train_input = WeightedSamplerSequence(
                train_metadata_csv=args.metadata_train,
                class_count=num_classes,
                snrs=TARGET_SNRS,
                batch_size=batch_size,
                epoch_size=args.epoch_size,
                weights=weights,
                shuffle_within_bucket=True,
                class_names=class_names,
            )
        else:
            ep_size = args.epoch_size or len(load_metadata_csv(args.metadata_train))
            if args.sampler_backend == 'tfdata':
                snr_list = None
                if args.snr_filter:
                    try:
                        candidate = [int(x.strip()) for x in args.snr_filter.split(',') if x.strip()]
                        snr_list = [s for s in candidate if s in TARGET_SNRS]
                    except Exception:
                        snr_list = None
                scope = args.uniform_scope
                print(f"[sampler-backend] Using tf.data with epoch_size={ep_size}, scope={scope}, snrs={snr_list or TARGET_SNRS}")
                train_input = _make_tfdata_from_sampler_draws(
                    train_meta_csv=args.metadata_train,
                    class_names=class_names,
                    weights=weights,
                    epoch_size=ep_size,
                    batch_size=batch_size,
                    image_size=image_size,
                    uniform_scope=scope,
                    snr_filter=snr_list,
                )
            else:
                print(f"[sampler-backend] Using tf.data-dir-class with epoch_size={ep_size}")
                train_input = _make_tfdata_class_uniform_from_dir(
                    train_dir=TRAIN_DIR,
                    class_names=class_names,
                    epoch_size=ep_size,
                    batch_size=batch_size,
                    image_size=image_size,
                )

        cb_list = [ckpt, csv, tb, reduce_lr, LrPrinter()]

        if args.debug_sampler and args.sampler_backend == 'sequence':
            try:
                X_dbg, y_dbg = train_input[0]
                unique, counts = np.unique(y_dbg, return_counts=True)
                print(f"[debug-sampler] y unique/counts: {list(zip(unique.tolist(), counts.tolist()))}")
                # Print a few random sample folder->label pairs from buckets
                import random as _rnd
                keys = list(train_input.buckets.keys())
                _rnd.shuffle(keys)
                for key in keys[:8]:
                    paths_k = train_input.buckets.get(key, [])
                    if not paths_k:
                        continue
                    p0 = paths_k[_rnd.randrange(len(paths_k))]
                    folder = os.path.basename(os.path.dirname(p0))
                    cid = train_input.class_name_to_id.get(folder, None) if train_input.class_name_to_id else None
                    print(f"[debug-sampler] example path: {p0} | folder={folder} -> label={cid}")
            except Exception as e:
                print(f"[debug-sampler] Failed to preview sampler batch: {e}")

        if args.debug_trainstep:
            debug_train_steps(model, train_input, num_classes, steps=5)
        if args.mode == 'adaptive':
            cb_list.append(ConfusionBySNRCallback(
                val_metadata_csv=args.metadata_val,
                weights_ref=weights,
                out_dir=RESULTS_DIR,
                beta=0.3,
                epsilon=0.02,
                max_cap=0.4,
                batch_size=batch_size,
                snrs=TARGET_SNRS,
                warmup_epochs=args.warmup_epochs,
                min_val_acc_for_updates=args.min_val_acc,
                class_names=class_names,
            ))
        if args.valprobe_batches:
            cb_list.append(ValProbeCallback(val_ds=val_ds, class_names=class_names, n_batches=int(args.valprobe_batches)))
        fit_kwargs['callbacks'] = cb_list

        # Dynamic resampling loop for tf.data backends if requested
        if args.sampler_backend in ('tfdata', 'tfdata-dir-class') and args.resample_each_epoch:
            print(f"[resample-loop] Enabled per-epoch resampling (backend={args.sampler_backend})")
            ep_size = args.epoch_size or len(load_metadata_csv(args.metadata_train))
            snr_list = None
            if args.snr_filter:
                try:
                    candidate = [int(x.strip()) for x in args.snr_filter.split(',') if x.strip()]
                    snr_list = [s for s in candidate if s in TARGET_SNRS]
                except Exception:
                    snr_list = None
            scope = args.uniform_scope
            for epoch in range(epochs):
                if args.sampler_backend == 'tfdata':
                    train_input = _make_tfdata_from_sampler_draws(
                        train_meta_csv=args.metadata_train,
                        class_names=class_names,
                        weights=weights,
                        epoch_size=ep_size,
                        batch_size=batch_size,
                        image_size=image_size,
                        uniform_scope=scope,
                        snr_filter=snr_list,
                    )
                else:
                    train_input = _make_tfdata_class_uniform_from_dir(
                        train_dir=TRAIN_DIR,
                        class_names=class_names,
                        epoch_size=ep_size,
                        batch_size=batch_size,
                        image_size=image_size,
                    )
                print(f"[resample-loop] Epoch {epoch+1}/{epochs}: built dataset with {ep_size} samples")
                model.fit(train_input,
                          validation_data=val_ds,
                          epochs=epoch+1,
                          initial_epoch=epoch,
                          callbacks=cb_list,
                          verbose=1)
        else:
            model.fit(train_input, **fit_kwargs)

    print(f"\nTraining complete. Best model saved to: {MODEL_OUT}")


if __name__ == '__main__':
    main()
