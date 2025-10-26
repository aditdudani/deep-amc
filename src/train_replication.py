import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint

from model_builder import build_googlenet_transfer

# --- Fidelity-first configuration (per Peng et al. 2018) ---
DATA_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = os.path.join('models', 'googlenet_fidelity_sgd.h5')

NUM_CLASSES = 8
IMAGE_SIZE = 224  # Paper uses 224x224 for GoogLeNet
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3  # SGD learning rate; adjust if unstable
MOMENTUM = 0.9


def make_datasets(train_dir, val_dir, image_size, batch_size):
    """Create tf.data datasets from directory structure with caching/prefetch."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names

    preprocess = tf.keras.applications.inception_v3.preprocess_input
    AUTOTUNE = tf.data.AUTOTUNE

    options = tf.data.Options()
    options.experimental_deterministic = False

    train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    cache_root = os.path.dirname(train_dir)
    train_cache_path = os.path.join(cache_root, 'train.cache')
    train_ds = train_ds.cache(train_cache_path)
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.prefetch(AUTOTUNE).with_options(options)

    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    val_cache_path = os.path.join(cache_root, 'val.cache')
    val_ds = val_ds.cache(val_cache_path)
    val_ds = val_ds.prefetch(AUTOTUNE).with_options(options)

    return train_ds, val_ds, class_names


def main():
    print("\n--- Fidelity-first training (single-phase, SGD, 224x224) ---\n")

    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed-precision policy:", mixed_precision.global_policy())

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    train_ds, val_ds, class_names = make_datasets(TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE)
    print(f"Detected classes ({len(class_names)}): {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"Warning: NUM_CLASSES={NUM_CLASSES} but dataset has {len(class_names)} classes.")

    # Build model: full network trainable from start (single-phase)
    model = build_googlenet_transfer(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        train_base=True
    )

    # Piecewise LR schedule in steps (epochs 0-14: 1e-3, 15-29: 1e-4, 30+: 1e-5)
    # We compute boundaries in steps to work robustly with the optimizer.
    try:
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    except Exception:
        steps_per_epoch = None

    if steps_per_epoch is None or steps_per_epoch < 0:
        # Fallback: no known cardinality; use fixed LR and emit a warning.
        print("Warning: could not determine steps_per_epoch; using fixed LR.")
        base_optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM)
    else:
        boundaries = [15 * steps_per_epoch, 30 * steps_per_epoch]
        values = [1e-3, 1e-4, 1e-5]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        base_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=MOMENTUM)

    optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )

    class LrPrinter(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            # If wrapped by LossScaleOptimizer, fetch inner optimizer's LR
            try:
                lr = self.model.optimizer._optimizer.learning_rate  # type: ignore[attr-defined]
            except Exception:
                pass
            current_lr = lr(self.model.optimizer.iterations) if callable(lr) else float(tf.keras.backend.get_value(lr))
            print(f"\n[Epoch {epoch+1}] Learning rate: {current_lr:.6g}")

    print('\nBeginning training...')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, LrPrinter()],
        verbose=1
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
