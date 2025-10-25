import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint

# Local import: reuse the project's model builder
from model_builder import build_googlenet_transfer

# --- Configuration (tweakable) ---
DATA_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = os.path.join('models', 'googlenet_finetuned.h5')

NUM_CLASSES = 8
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4


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

    train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    # Cache in memory after first epoch to eliminate disk I/O for later epochs
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(AUTOTUNE)

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

    # Build model and enable fine-tuning of the base model
    model = build_googlenet_transfer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                     num_classes=NUM_CLASSES,
                                     train_base=True)

    # Optimizer: lower learning rate for fine-tuning pretrained weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Wrap with LossScaleOptimizer for mixed-precision stability
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

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

    print('\nBeginning training...')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )

    print(f"\nTraining complete. Best model (by val_accuracy) saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
