import os
import sys
import json
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

"""Train SqueezeNet with adaptive sampler.
Run as: python src/adaptive_sampling/train_squeezenet_sampler.py
"""

# Ensure src/ is on sys.path so `common` imports work when run as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.squeezenet import build_squeezenet_v11
from common.config import IMAGE_SIZE, TARGET_SNRS
from adaptive_sampling.sampler import WeightedSamplerSequence, load_metadata_csv, init_uniform_weights
from adaptive_sampling.callbacks_confusion_snr import ConfusionBySNRCallback


# --------------------
# Config (no CLI)
# --------------------
DATA_DIR = os.path.join('data', 'processed')
TRAIN_META = os.path.join(DATA_DIR, 'metadata_train.csv')
VAL_META = os.path.join(DATA_DIR, 'metadata_val.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-2
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join('results', 'adaptive_sampling', RUN_TAG)
MODEL_OUT = os.path.join('models', f'squeezenet_sampler_{RUN_TAG}.h5')
LOG_CSV = os.path.join(RESULTS_DIR, 'squeezenet_sampler_train_log.csv')
TB_LOGDIR = os.path.join(RESULTS_DIR, 'logs')


class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        current_lr = lr(self.model.optimizer.iterations) if callable(lr) else float(tf.keras.backend.get_value(lr))
        print(f"\n[Epoch {epoch+1}] Learning rate: {current_lr:.6g}")


def _determine_num_classes(train_dir: str) -> int:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()
    return len(classes)


def main():
    tf.keras.mixed_precision.set_global_policy('float32')

    # Preconditions
    if not os.path.exists(TRAIN_META) or not os.path.exists(VAL_META):
        raise FileNotFoundError(
            f"Missing metadata CSVs. Please run src/adaptive_sampling/dataset_metadata.py to generate:\n  {TRAIN_META}\n  {VAL_META}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TB_LOGDIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    num_classes = _determine_num_classes(TRAIN_DIR)

    # Initialize weights uniform
    weights = init_uniform_weights(num_classes, TARGET_SNRS)
    with open(os.path.join(RESULTS_DIR, 'weights_epoch0.json'), 'w') as f:
        json.dump({'epoch': 0, 'weights': weights.tolist(), 'snrs': TARGET_SNRS}, f, indent=2)

    # Determine epoch size as number of training samples for parity with baseline
    train_records = load_metadata_csv(TRAIN_META)
    epoch_size = len(train_records)

    train_seq = WeightedSamplerSequence(
        train_metadata_csv=TRAIN_META,
        class_count=num_classes,
        snrs=TARGET_SNRS,
        batch_size=BATCH_SIZE,
        epoch_size=epoch_size,
        weights=weights,
        shuffle_within_bucket=True,
        seed=1234,
    )

    # Validation for Keras metrics (class-only accuracy)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Build model
    model = build_squeezenet_v11(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=num_classes, dropout_rate=0.0)
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
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

    confusion_cb = ConfusionBySNRCallback(
        val_metadata_csv=VAL_META,
        weights_ref=weights,
        out_dir=RESULTS_DIR,
        beta=0.1,         # milder updates; safer vs baseline
        epsilon=0.01,     # smaller floor to reduce perturbation
        max_cap=0.25,     # cap a single bucket to 25%
        batch_size=BATCH_SIZE,
        snrs=TARGET_SNRS,
        warmup_epochs=3,
    )

    print("\n--- Training SqueezeNet v1.1 with adaptive sampler ---")
    model.fit(
        train_seq,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, csv, tb, reduce_lr, LrPrinter(), confusion_cb],
        verbose=1,
        workers=4,
        use_multiprocessing=False,
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_OUT}")


if __name__ == '__main__':
    main()
