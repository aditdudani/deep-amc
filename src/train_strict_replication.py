import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint


# --- Config (no reprocessing; reuse existing images) ---
DATA_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = os.path.join('models', 'strict_googlenet_proxy_incv3.h5')

NUM_CLASSES = 8
IMAGE_SIZE = 224
BATCH_SIZE = 64

# Training schedule
WARMUP_EPOCHS = 5  # head-only (frozen base) with Adam
TOTAL_EPOCHS = 45  # total epochs (warm-up + fine-tune)

# Optimizer params
HEAD_LR = 1e-3
SGD_MOMENTUM = 0.9


def iv3_preprocess(inputs):
    """InceptionV3 native preprocessing: scales RGB inputs from [0,255] to [-1,1]."""
    return tf.keras.applications.inception_v3.preprocess_input(inputs)


def build_inceptionv3_with_native_preprocess(input_shape=(224, 224, 3), num_classes=8, freeze_bn=False):
    inputs = layers.Input(shape=input_shape, dtype='float32')
    x = layers.Lambda(iv3_preprocess, name='iv3_preprocess')(inputs)

    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
    )

    if freeze_bn:
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    y = layers.GlobalAveragePooling2D(name='avg_pool')(base.output)
    y = layers.Dropout(0.4)(y)
    # Keep logits in float32 for numerical stability
    outputs = layers.Dense(num_classes, activation=None, dtype='float32', name='logits')(y)

    model = models.Model(inputs=inputs, outputs=outputs, name='inceptionv3_native_head')
    return model, base


def make_datasets(train_dir, val_dir, image_size, batch_size):
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
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    # Do NOT apply Keras preprocess here; model has Caffe preprocessing layer.
    train_ds = train_ds.prefetch(AUTOTUNE).with_options(options)
    val_ds = val_ds.prefetch(AUTOTUNE).with_options(options)

    return train_ds, val_ds


class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        current_lr = lr(self.model.optimizer.iterations) if callable(lr) else float(tf.keras.backend.get_value(lr))
        print(f"\n[Epoch {epoch+1}] Learning rate: {current_lr:.6g}")


def main():
    # Strict float32 training for fidelity
    tf.keras.mixed_precision.set_global_policy('float32')

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Expected directories: {TRAIN_DIR} and {VAL_DIR}")

    train_ds, val_ds = make_datasets(TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE)

    # Build model with BN frozen initially (helps stability when fine-tuning later)
    model, base = build_inceptionv3_with_native_preprocess(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        freeze_bn=True,
    )

    # Warm-up: train head only
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    print("\n--- Warm-up: training head only (frozen base) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=[LrPrinter()],
        verbose=1,
    )

    # Fine-tune: unfreeze entire base (keep BN frozen via freeze_bn flag)
    base.trainable = True
    # Re-apply BN freezing explicitly (in case state changed)
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Step LR schedule (0-14: 1e-3, 15-29: 1e-4, 30+: 1e-5) for the fine-tune phase length
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    boundaries = [15 * steps_per_epoch, 30 * steps_per_epoch]
    values = [1e-3, 1e-4, 1e-5]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=SGD_MOMENTUM, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    ckpt = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max',
    )

    remaining_epochs = max(TOTAL_EPOCHS - WARMUP_EPOCHS, 1)
    print("\n--- Fine-tune: unfreezing base with SGD + step LR ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=remaining_epochs,
        callbacks=[ckpt, LrPrinter()],
        verbose=1,
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
