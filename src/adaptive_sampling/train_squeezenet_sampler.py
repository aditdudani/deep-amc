import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np

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
    p.add_argument('--image-size', type=int, default=IMAGE_SIZE)
    p.add_argument('--warmup-epochs', type=int, default=3, help='Adaptive: epochs to skip updates')
    p.add_argument('--min-val-acc', type=float, default=0.15, help='Adaptive: min val_acc to allow updates')
    p.add_argument('--metadata-train', type=str, default=TRAIN_META_CSV)
    p.add_argument('--metadata-val', type=str, default=VAL_META_CSV)
    p.add_argument('--verify-metadata', action='store_true', help='Quickly verify metadata paths/SNRs before sampler/adaptive runs')
    p.add_argument('--debug-sampler', action='store_true', help='Dump a preview batch from sampler and basic stats')
    p.add_argument('--epoch-size', type=int, default=None, help='Sampler: number of samples per epoch (for quick tests)')
    p.add_argument('--sampler-backend', type=str, default='sequence', choices=['sequence', 'tfdata'],
                   help='Use Keras Sequence or tf.data backend for sampler modes')
    p.add_argument('--debug-trainstep', action='store_true', help='Run a few train_on_batch steps to verify learning')
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
                                    epoch_size: int, batch_size: int, image_size: int):
    # Build buckets consistent with WeightedSamplerSequence
    records = load_metadata_csv(train_meta_csv)
    class_name_to_id = {name: i for i, name in enumerate(class_names)}
    buckets, snr_to_idx = build_buckets(records, TARGET_SNRS, class_name_to_id)

    # Pre-sample a list of (path, label) for this epoch
    flat = weights.flatten()
    num_classes = len(class_names)
    num_snrs = len(TARGET_SNRS)
    paths = []
    labels = []
    rng = np.random.default_rng()
    for _ in range(epoch_size):
        choice = rng.choice(len(flat), p=flat)
        cid = choice // num_snrs
        snr_idx = choice % num_snrs
        snr = TARGET_SNRS[snr_idx]
        key = (cid, snr)
        lst = buckets.get(key, [])
        if not lst:
            # fallback within class
            class_paths = []
            for (c, s), l in buckets.items():
                if c == cid and l:
                    class_paths.extend(l)
            if class_paths:
                pth = class_paths[rng.integers(0, len(class_paths))]
            else:
                # worst-case fallback: random record path
                pth = records[rng.integers(0, len(records))][0]
        else:
            pth = lst[rng.integers(0, len(lst))]
        folder = os.path.basename(os.path.dirname(pth))
        label = class_name_to_id.get(folder, cid)
        paths.append(pth)
        labels.append(label)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load_fn(pth, y):
        img_bytes = tf.io.read_file(pth)
        img = tf.io.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32)
        return img, y
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.shuffle(min(10000, epoch_size)).map(_load_fn, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return ds


def _first_weight_vector(model: tf.keras.Model):
    for w in model.trainable_weights:
        if 'kernel' in w.name or 'weights' in w.name:
            return tf.reshape(w, [-1])
    return None


def _debug_train_steps(model: tf.keras.Model, input_source, num_classes: int, steps: int = 5):
    print("[debug-trainstep] Running a few train_on_batch steps...")
    w0 = _first_weight_vector(model)
    w0_val = tf.identity(w0) if w0 is not None else None
    get_batch = None
    if isinstance(input_source, tf.keras.utils.Sequence):
        idx = 0
        def _gb():
            nonlocal idx
            X, y = input_source[idx]
            idx = (idx + 1) % len(input_source)
            return X, y
        get_batch = _gb
    else:
        it = iter(input_source)
        def _gb():
            X, y = next(it)
            return X.numpy() if hasattr(X, 'numpy') else X, y.numpy() if hasattr(y, 'numpy') else y
        get_batch = _gb

    for i in range(steps):
        X, y = get_batch()
        y_min, y_max = int(np.min(y)), int(np.max(y))
        if y_min < 0 or y_max >= num_classes:
            print(f"[debug-trainstep] Label out of range: min={y_min}, max={y_max}, num_classes={num_classes}")
        loss, acc = model.train_on_batch(X, y, return_dict=False)
        print(f"[debug-trainstep] step {i+1}: loss={loss:.4f}, acc={acc:.4f}")
    if w0 is not None:
        w1 = _first_weight_vector(model)
        delta = tf.norm(w1 - w0_val).numpy()
        print(f"[debug-trainstep] weight L2 delta after {steps} steps: {delta:.6f}")



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
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
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
            print(f"[sampler-backend] Using tf.data with epoch_size={ep_size}")
            train_input = _make_tfdata_from_sampler_draws(
                train_meta_csv=args.metadata_train,
                class_names=class_names,
                weights=weights,
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
                # Print a few sample folder->label pairs
                shown = 0
                for key, paths in train_input.buckets.items():
                    if not paths:
                        continue
                    p0 = paths[0]
                    folder = os.path.basename(os.path.dirname(p0))
                    cid = train_input.class_name_to_id.get(folder, None) if train_input.class_name_to_id else None
                    print(f"[debug-sampler] example path: {p0} | folder={folder} -> label={cid}")
                    shown += 1
                    if shown >= 8:
                        break
            except Exception as e:
                print(f"[debug-sampler] Failed to preview sampler batch: {e}")

        if args.debug_trainstep:
            _debug_train_steps(model, train_input, num_classes, steps=5)
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
        fit_kwargs['callbacks'] = cb_list
        model.fit(train_input, **fit_kwargs)

    print(f"\nTraining complete. Best model saved to: {MODEL_OUT}")


if __name__ == '__main__':
    main()
