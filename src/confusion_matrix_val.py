import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# --------- Config (no CLI) ---------
VAL_DIR = os.path.join('data', 'processed', 'validation')
MODEL_PATH = os.path.join('models', 'squeezenet_v11_rmsprop.h5')
IMAGE_SIZE = 224
BATCH_SIZE = 64
OUT_PNG = os.path.join('results', 'confusion_matrix_val.png')


def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    os.makedirs('results', exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    class_names = ds.class_names
    num_classes = len(class_names)
    print(f"Validation classes ({num_classes}): {class_names}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_true_all = []
    y_pred_all = []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        y_true_all.append(batch_y.numpy())
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    acc = float((y_true == y_pred).mean())
    print(f"Validation accuracy: {acc:.4f} (n={len(y_true)})")

    cm = _compute_confusion_matrix(y_true, y_pred, num_classes)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Confusion Matrix (Validation)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(num_classes), labels=class_names)
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"Saved confusion matrix: {OUT_PNG}")


if __name__ == '__main__':
    main()
