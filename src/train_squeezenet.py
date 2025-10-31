import os
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from squeezenet import build_squeezenet_v11


# --------------------
# Config (no CLI)
# --------------------
DATA_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 40  # extend now that learning is confirmed; ReduceLROnPlateau will anneal LR
LEARNING_RATE = 1e-2  # keep LR that worked in the successful probe
MODEL_OUT = os.path.join('models', 'squeezenet_v11_rmsprop.h5')
LOG_CSV = os.path.join('results', 'squeezenet_train_log.csv')
TB_LOGDIR = os.path.join('results', 'logs', 'squeezenet')


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

    # Prefetch for performance; SqueezeNet model handles normalization (Rescaling layer)
    train_ds = train_ds.prefetch(AUTOTUNE).with_options(options)
    val_ds = val_ds.prefetch(AUTOTUNE).with_options(options)
    return train_ds, val_ds
class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        current_lr = lr(self.model.optimizer.iterations) if callable(lr) else float(tf.keras.backend.get_value(lr))
        print(f"\n[Epoch {epoch+1}] Learning rate: {current_lr:.6g}")


def main():
    # Keep it deterministic yet simple
    tf.keras.mixed_precision.set_global_policy('float32')

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Expected directories: {TRAIN_DIR} and {VAL_DIR}")

    train_ds, val_ds = make_datasets(TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE)

    # Infer number of classes from directories
    class_names = train_ds.class_names if hasattr(train_ds, 'class_names') else None
    if class_names is None:
        class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    model = build_squeezenet_v11(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=num_classes, dropout_rate=0.2)

    # Optimizer choice: RMSprop (stable for this small net). Switch to SGD if desired.
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    os.makedirs(TB_LOGDIR, exist_ok=True)

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

    print("\n--- Training SqueezeNet v1.1 (RMSprop) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, csv, tb, reduce_lr, LrPrinter()],
        verbose=1,
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_OUT}")

    # Also export a SavedModel for later conversion/analysis
    savedmodel_dir = os.path.splitext(MODEL_OUT)[0] + "_savedmodel"
    tf.saved_model.save(model, savedmodel_dir)
    print(f"SavedModel exported to: {savedmodel_dir}")


if __name__ == '__main__':
    main()
