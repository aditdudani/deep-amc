"""
Clean evaluation script - evaluates models on validation split ONLY using pre-generated PNGs.

No HDF5 contamination. Per-class per-SNR accuracy matrix (8 classes × 6 SNRs).

Usage:
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path <path> --model-name <name>
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Tuple

# Suppress TensorFlow verbose logging BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
plt.set_loglevel('error')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE
from callbacks_confusion_snr import load_validation_metadata


def eval_validation_clean(
    model_path: str,
    metadata_val_csv: str,
    output_dir: str,
    model_name: str,
    batch_size: int = 64
):
    """Evaluate model on validation split using metadata CSV (clean, uncontaminated)."""

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✓ Loaded model: {model_path}")

    # Load metadata
    if not os.path.exists(metadata_val_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_val_csv}")
    val_items = load_validation_metadata(metadata_val_csv)
    print(f"✓ Loaded {len(val_items):,} validation samples")

    # Infer class order from directory structure
    inferred_classes = []
    for file_path, class_id, snr in val_items:
        folder = os.path.basename(os.path.dirname(file_path))
        if folder not in inferred_classes:
            inferred_classes.append(folder)
    inferred_classes.sort()
    class_to_id = {c: i for i, c in enumerate(inferred_classes)}
    print(f"✓ Classes ({len(inferred_classes)}): {', '.join(inferred_classes)}")

    # Group by SNR
    by_snr: Dict[int, List[Tuple[str, int, int]]] = {s: [] for s in TARGET_SNRS}
    for file_path, class_id, snr in val_items:
        if snr in by_snr:
            by_snr[snr].append((file_path, class_id, snr))

    # Evaluate per SNR
    num_classes = len(inferred_classes)
    per_class_snr_acc = np.zeros((num_classes, len(TARGET_SNRS)))
    acc_by_snr = {}
    counts_by_snr = {}
    per_class_total = {c: 0 for c in inferred_classes}
    per_class_correct = {c: 0 for c in inferred_classes}
    overall_correct = 0
    overall_total = 0

    print(f"\n{'─'*70}")
    print(f"Evaluating by SNR")
    print(f"{'─'*70}")

    for snr_idx, snr in enumerate(sorted(TARGET_SNRS)):
        items = by_snr[snr]
        if not items:
            continue

        # Load images and predict in batches
        X_batch = []
        y_true = []

        for file_path, class_id, _ in items:
            if not os.path.exists(file_path):
                continue
            try:
                img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                arr = tf.keras.utils.img_to_array(img)
                X_batch.append(arr)
                folder = os.path.basename(os.path.dirname(file_path))
                y_true.append(class_to_id[folder])
            except Exception:
                continue

        if not X_batch:
            continue

        X_batch = np.array(X_batch, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int64)

        # Predict
        probs = model.predict(X_batch, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        # Per-class accuracy for this SNR
        for class_idx, class_name in enumerate(inferred_classes):
            mask = y_true == class_idx
            if mask.sum() > 0:
                acc = (y_pred[mask] == class_idx).sum() / mask.sum()
                per_class_snr_acc[class_idx, snr_idx] = acc
                per_class_total[class_name] += int(mask.sum())
                per_class_correct[class_name] += int((y_pred[mask] == class_idx).sum())

        # Overall accuracy for SNR
        correct_snr = (y_pred == y_true).sum()
        total_snr = len(y_true)
        acc = correct_snr / total_snr if total_snr > 0 else 0.0
        acc_by_snr[snr] = float(acc)
        counts_by_snr[snr] = int(total_snr)
        overall_correct += correct_snr
        overall_total += total_snr

        print(f"  SNR {snr:>2} dB → {acc*100:6.2f}% ({total_snr:,} samples)")

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print(f"\n{'─'*70}")
    print(f"Overall Accuracy: {overall_acc*100:.2f}% ({overall_total:,} samples)")
    print(f"{'─'*70}")

    # Per-class overall accuracy
    print(f"\nPer-Class Accuracy:")
    print(f"{'─'*70}")
    per_class_overall = {}
    for class_name in inferred_classes:
        if per_class_total[class_name] > 0:
            acc = per_class_correct[class_name] / per_class_total[class_name]
            per_class_overall[class_name] = float(acc)
            print(f"  {class_name:>8} → {acc*100:6.2f}% ({per_class_correct[class_name]:,}/{per_class_total[class_name]:,})")
        else:
            per_class_overall[class_name] = 0.0

    # Save JSON with complete results
    result = {
        "model_name": model_name,
        "model_path": model_path,
        "metadata_csv": metadata_val_csv,
        "accuracy_by_snr": {int(k): v for k, v in acc_by_snr.items()},
        "counts_by_snr": {int(k): v for k, v in counts_by_snr.items()},
        "overall_accuracy": float(overall_acc),
        "overall_count": int(overall_total),
        "per_class_snr_accuracy": per_class_snr_acc.tolist(),
        "per_class_overall_accuracy": per_class_overall,
        "classes": inferred_classes,
        "snrs": TARGET_SNRS,
    }

    json_path = os.path.join(output_dir, f"eval_{model_name}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved JSON: {json_path}")

    # Plot accuracy by SNR
    snrs = sorted(acc_by_snr.keys())
    accs = [acc_by_snr[s] * 100 for s in snrs]

    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accs, marker='o', linewidth=2.5, markersize=8, label=model_name, color='tab:blue')
    plt.axhline(y=overall_acc*100, color='gray', linestyle='--', alpha=0.6, linewidth=1.5,
                label=f'Overall ({overall_acc*100:.2f}%)')
    plt.title(f'Validation Accuracy by SNR: {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(snrs)
    plt.ylim(0, 105)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"eval_{model_name}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {png_path}")
    plt.close()

    # Plot per-class heatmap (8 classes × 6 SNRs)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(per_class_snr_acc * 100, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(TARGET_SNRS)))
    ax.set_xticklabels([f'{s}dB' for s in TARGET_SNRS])
    ax.set_yticks(range(len(inferred_classes)))
    ax.set_yticklabels(inferred_classes)
    ax.set_xlabel('SNR', fontsize=11)
    ax.set_ylabel('Class', fontsize=11)
    ax.set_title(f'Per-Class Per-SNR Accuracy (%): {model_name}', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(inferred_classes)):
        for j in range(len(TARGET_SNRS)):
            text = ax.text(j, i, f'{per_class_snr_acc[i, j]*100:.0f}',
                          ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', fontsize=11)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, f"eval_{model_name}_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmap: {heatmap_path}")
    plt.close()

    return result


def parse_args():
    p = argparse.ArgumentParser(
        description='Clean validation evaluation (no HDF5 contamination)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path results_local/phase2_matrix/model_config_A.keras \
    --model-name "Config_A_Baseline"
        """
    )
    p.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    p.add_argument('--model-name', type=str, required=True, help='Name for results')
    p.add_argument('--metadata-val', type=str, default='data/processed_g/metadata_val.csv',
                   help='Path to validation metadata CSV')
    p.add_argument('--output-dir', type=str, default='results_local/phase3_clean_eval',
                   help='Directory to save results')
    p.add_argument('--batch-size', type=int, default=64, help='Prediction batch size')
    return p.parse_args()


def main():
    args = parse_args()
    eval_validation_clean(
        model_path=args.model_path,
        metadata_val_csv=args.metadata_val,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
