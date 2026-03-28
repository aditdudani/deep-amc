"""
Clean evaluation script - evaluates models on validation split ONLY using pre-generated PNGs.

No HDF5 contamination. Per-class per-SNR accuracy matrix (8 classes × 6 SNRs).

Usage:
  # Config A baseline
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path results_local/phase2_matrix/model_config_A.keras \
    --model-name "Config_A_Baseline"

  # Config G baseline
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path results_local/phase2_matrix/model_config_G.keras \
    --model-name "Config_G_Baseline"

  # Adaptive Config G
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path models/squeezenet_sampler_g_20260224_155039.h5 \
    --model-name "Config_G_Adaptive"
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    """
    Evaluate model using validation metadata CSV (clean, no HDF5 contamination).

    Args:
        model_path: Path to trained Keras model
        metadata_val_csv: Path to validation metadata CSV
        output_dir: Directory to save results
        model_name: Descriptive name for the model (used in outputs)
        batch_size: Prediction batch size
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"[eval] Loaded model: {model_path}")
    print(f"[eval] Model input shape: {model.input_shape}")

    # Load metadata
    if not os.path.exists(metadata_val_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_val_csv}")
    val_items = load_validation_metadata(metadata_val_csv)
    print(f"[eval] Loaded {len(val_items)} validation items from {metadata_val_csv}")

    # Infer class order from directory structure
    inferred_classes = []
    for file_path, class_id, snr in val_items:
        folder = os.path.basename(os.path.dirname(file_path))
        if folder not in inferred_classes:
            inferred_classes.append(folder)
    inferred_classes.sort()
    class_to_id = {c: i for i, c in enumerate(inferred_classes)}
    print(f"[eval] Classes ({len(inferred_classes)}): {inferred_classes}")

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

    print(f"\n--- Evaluation by SNR ---")
    for snr_idx, snr in enumerate(sorted(TARGET_SNRS)):
        items = by_snr[snr]
        if not items:
            print(f"[eval] SNR {snr:>2} dB: No items")
            continue

        # Load images and predict in batches
        X_batch = []
        y_true = []
        metadata_items = []

        for file_path, class_id, _ in items:
            if not os.path.exists(file_path):
                print(f"[warn] File not found: {file_path}, skipping")
                continue

            try:
                img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                arr = tf.keras.utils.img_to_array(img)
                X_batch.append(arr)
                folder = os.path.basename(os.path.dirname(file_path))
                y_true.append(class_to_id[folder])
                metadata_items.append((file_path, folder))
            except Exception as e:
                print(f"[warn] Error loading {file_path}: {e}")
                continue

        if not X_batch:
            print(f"[eval] SNR {snr:>2} dB: No valid items")
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

        print(f"[eval] SNR {snr:>2} dB -> accuracy: {acc*100:6.2f}% (n={total_snr})")

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print(f"\n[eval] Overall accuracy: {overall_acc*100:.2f}% (n={overall_total})")

    # Per-class overall accuracy
    print(f"\n--- Per-Class Overall Accuracy ---")
    per_class_overall = {}
    for class_name in inferred_classes:
        if per_class_total[class_name] > 0:
            acc = per_class_correct[class_name] / per_class_total[class_name]
            per_class_overall[class_name] = float(acc)
            print(f"{class_name:>8} -> {acc*100:6.2f}% ({per_class_correct[class_name]}/{per_class_total[class_name]})")
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
    print(f"\n[save] JSON: {json_path}")

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
    plt.savefig(png_path, dpi=150)
    print(f"[save] Plot: {png_path}")
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
    plt.savefig(heatmap_path, dpi=150)
    print(f"[save] Heatmap: {heatmap_path}")
    plt.close()

    return result


def parse_args():
    p = argparse.ArgumentParser(
        description='Clean validation evaluation (no HDF5 contamination)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Config A baseline
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path results_local/phase2_matrix/model_config_A.keras \
    --model-name "Config_A_Baseline"

  # Config G baseline
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path results_local/phase2_matrix/model_config_G.keras \
    --model-name "Config_G_Baseline"

  # Adaptive Config G
  python src/adaptive_sampling_g/eval_validation_clean.py \
    --model-path models/squeezenet_sampler_g_20260224_155039.h5 \
    --model-name "Config_G_Adaptive"
        """
    )
    p.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    p.add_argument('--model-name', type=str, required=True, help='Name for results (e.g. Config_A_Baseline)')
    p.add_argument('--metadata-val', type=str, default='data/processed_g/metadata_val.csv',
                   help='Path to validation metadata CSV')
    p.add_argument('--output-dir', type=str, default='results_local/phase3_clean_eval',
                   help='Directory to save evaluation results')
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
