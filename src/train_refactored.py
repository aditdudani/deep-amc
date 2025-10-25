import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# Local import: reuse the project's model builder
from model_builder import build_googlenet_transfer

# --- Configuration (tweakable) ---
DATA_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = os.path.join('models', 'googlenet_finetuned.h5')

NUM_CLASSES = 8
IMAGE_SIZE = 299  # Align with InceptionV3 pretraining
BATCH_SIZE = 48   # Slightly reduced for 299x299 tensors
EPOCHS = 50       # Total epochs (warm-up + fine-tune)

# Two-phase training configuration
WARMUP_EPOCHS = 5           # Train head-only with base frozen
WARMUP_LR = 1e-3            # Higher LR for classifier head
FINETUNE_LR = 1e-5          # Low LR for fine-tuning base
MOMENTUM = 0.9


def make_datasets(train_dir, val_dir, image_size, batch_size):
    """Create optimized tf.data.Dataset objects from directory structure.

    Assumes subdirectories in `train_dir` and `val_dir` correspond to class names.
    Returns (train_ds, val_ds, class_names)
    """
    # Use Keras utility to create datasets from folders. Labels inferred from
    # subdirectory names. label_mode='int' yields integer labels (suitable for
    # SparseCategoricalCrossentropy).
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

    # Preprocessing: use the InceptionV3 preprocessing function which expects
    # images in RGB with pixel values in [0,255]. The function will scale them
    # appropriately for the network.
    preprocess = tf.keras.applications.inception_v3.preprocess_input

    AUTOTUNE = tf.data.AUTOTUNE

    # Enable non-deterministic order for increased throughput
    options = tf.data.Options()
    options.experimental_deterministic = False

    train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    # Use disk-backed cache to avoid excessive RAM usage on large datasets
    cache_root = os.path.dirname(train_dir)
    train_cache_path = os.path.join(cache_root, 'train.cache')
    train_ds = train_ds.cache(train_cache_path)
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.prefetch(AUTOTUNE)
    train_ds = train_ds.with_options(options)

    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    val_cache_path = os.path.join(cache_root, 'val.cache')
    val_ds = val_ds.cache(val_cache_path)
    val_ds = val_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.with_options(options)

    return train_ds, val_ds, class_names


def main():
    print("\n--- Starting refactored training script (train_refactored.py) ---\n")

    # Enable mixed precision for speed on supporting GPUs
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed-precision policy set to:", mixed_precision.global_policy())

    # Sanity checks for data directories
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    # Create optimized datasets
    train_ds, val_ds, class_names = make_datasets(TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE)

    print(f"Detected classes ({len(class_names)}): {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"Warning: NUM_CLASSES={NUM_CLASSES} but dataset contains {len(class_names)} classes.")

    # Phase A: Warm-up head (freeze base)
    print("\nPhase A: Warm-up classifier head (base frozen)")
    model = build_googlenet_transfer(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        train_base=False
    )

    warmup_opt = tf.keras.optimizers.SGD(learning_rate=WARMUP_LR, momentum=MOMENTUM)
    warmup_opt = mixed_precision.LossScaleOptimizer(warmup_opt)
    model.compile(
        optimizer=warmup_opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs('results', exist_ok=True)
    csv_logger = CSVLogger(os.path.join('results', 'training_log.csv'), append=True)
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=[checkpoint, csv_logger],
        verbose=1
    )

    # Phase B: Fine-tune base (unfreeze base; keep BatchNorm layers frozen)
    print("\nPhase B: Fine-tune base (unfreeze backbone with BN frozen)")
    base = None
    try:
        base = model.get_layer('inception_v3')
    except Exception:
        base = None
    if base is not None:
        base.trainable = True
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
    else:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    finetune_opt = tf.keras.optimizers.SGD(learning_rate=FINETUNE_LR, momentum=MOMENTUM)
    finetune_opt = mixed_precision.LossScaleOptimizer(finetune_opt)
    model.compile(
        optimizer=finetune_opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32
    )

    lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS - WARMUP_EPOCHS,
        callbacks=[checkpoint, csv_logger, lr_plateau],
        verbose=1
    )

    print(f"\nTraining complete. Best model (by val_accuracy) saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
