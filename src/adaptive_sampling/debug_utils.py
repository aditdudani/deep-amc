import numpy as np
import tensorflow as tf


def _first_weight_vector(model: tf.keras.Model):
    for w in model.trainable_weights:
        if 'kernel' in w.name or 'weights' in w.name:
            return tf.reshape(w, [-1])
    return None


def debug_train_steps(model: tf.keras.Model, input_source, num_classes: int, steps: int = 5):
    print("[debug-trainstep] Running a few train_on_batch steps...")
    w0 = _first_weight_vector(model)
    w0_val = tf.identity(w0) if w0 is not None else None
    if isinstance(input_source, tf.keras.utils.Sequence):
        idx = 0
        def _gb():
            nonlocal idx
            X, y = input_source[idx]
            idx = (idx + 1) % len(input_source)
            return X, y
    else:
        it = iter(input_source)
        def _gb():
            X, y = next(it)
            return X.numpy() if hasattr(X, 'numpy') else X, y.numpy() if hasattr(y, 'numpy') else y

    for i in range(steps):
        X, y = _gb()
        y_min, y_max = int(np.min(y)), int(np.max(y))
        if y_min < 0 or y_max >= num_classes:
            print(f"[debug-trainstep] Label out of range: min={y_min}, max={y_max}, num_classes={num_classes}")
        loss, acc = model.train_on_batch(X, y, return_dict=False)
        print(f"[debug-trainstep] step {i+1}: loss={loss:.4f}, acc={acc:.4f}")
    if w0 is not None:
        w1 = _first_weight_vector(model)
        delta = tf.norm(w1 - w0_val).numpy()
        print(f"[debug-trainstep] weight L2 delta after {steps} steps: {delta:.6f}")


def val_probe(model: tf.keras.Model, val_ds, class_names, n_batches: int = 10, tag: str = ''):
    preds_hist = np.zeros((len(class_names),), dtype=np.int64)
    true_hist = np.zeros((len(class_names),), dtype=np.int64)
    correct = 0
    total = 0
    for i, (x, y) in enumerate(val_ds):
        logits = model.predict(x, verbose=0)
        p = np.argmax(logits, axis=1)
        y_np = y.numpy() if hasattr(y, 'numpy') else np.array(y)
        for cls in range(len(class_names)):
            preds_hist[cls] += int(np.sum(p == cls))
            true_hist[cls] += int(np.sum(y_np == cls))
        correct += int(np.sum(p == y_np))
        total += int(y_np.shape[0])
        if (i + 1) >= n_batches:
            break
    acc = (correct / total) if total else 0.0
    top_pred = int(np.argmax(preds_hist)) if preds_hist.sum() > 0 else -1
    print(f"[valprobe{':' + tag if tag else ''}] batches={n_batches} acc={acc:.4f} total={total} top_pred={top_pred} ({class_names[top_pred] if top_pred>=0 else 'n/a'})")
    print(f"[valprobe] preds_hist={preds_hist.tolist()} true_hist={true_hist.tolist()}")


class ValProbeCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, class_names, n_batches: int):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.n_batches = n_batches
    def on_epoch_end(self, epoch, logs=None):
        val_probe(self.model, self.val_ds, self.class_names, self.n_batches, tag=f'epoch{epoch+1}')
