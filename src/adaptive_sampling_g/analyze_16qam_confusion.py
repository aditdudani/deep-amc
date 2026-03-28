#!/usr/bin/env python3
"""
Analyze 16QAM confusion patterns, especially at 0dB SNR.

Focuses on understanding what classes 16QAM is being misclassified as,
to inform hyperparameter tuning decisions.
"""

import os
import sys
import json
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.config import TARGET_MODS, TARGET_SNRS, IMAGE_SIZE
from callbacks_confusion_snr import load_validation_metadata, _predict_in_batches, _confusion_matrix


def analyze_model_confusion(model_path: str, model_name: str, output_dir: str = 'results_local/phase3_clean_eval'):
    """Analyze confusion patterns for a model, focusing on 16QAM at 0dB."""

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✓ Loaded model: {model_path}")

    # Load validation metadata
    val_items = load_validation_metadata('data/processed_g/metadata_val.csv')
    print(f"✓ Loaded {len(val_items):,} validation samples")

    # Infer class order
    inferred_classes = []
    for file_path, class_id, snr in val_items:
        folder = os.path.basename(os.path.dirname(file_path))
        if folder not in inferred_classes:
            inferred_classes.append(folder)
    inferred_classes.sort()
    class_to_id = {c: i for i, c in enumerate(inferred_classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    print(f"✓ Classes ({len(inferred_classes)}): {', '.join(inferred_classes)}")

    # Group by class and SNR
    by_class_snr = {}
    for file_path, class_id, snr in val_items:
        folder = os.path.basename(os.path.dirname(file_path))
        class_name = folder
        key = (class_name, snr)
        if key not in by_class_snr:
            by_class_snr[key] = []
        by_class_snr[key].append((file_path, class_id, snr))

    print(f"\n{'='*80}")
    print(f"{'16QAM CONFUSION ANALYSIS':^80}")
    print(f"{'='*80}")

    # Focus on 16QAM
    target_class = '16QAM'
    target_class_id = class_to_id[target_class]

    for snr in sorted(TARGET_SNRS):
        key = (target_class, snr)
        items = by_class_snr.get(key, [])

        if not items:
            print(f"\n[SNR {snr:>2} dB] No {target_class} samples found")
            continue

        print(f"\n[SNR {snr:>2} dB] {target_class} ({len(items)} samples)")
        print(f"{'─'*80}")

        # Load and predict
        X_batch = []
        y_true = []
        for file_path, class_id, _ in items:
            if not os.path.exists(file_path):
                continue
            try:
                img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                arr = tf.keras.utils.img_to_array(img)
                X_batch.append(arr)
                y_true.append(class_id)
            except Exception as e:
                print(f"  ⚠ Error loading {file_path}: {e}")
                continue

        if not X_batch:
            continue

        X_batch = np.array(X_batch, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int64)

        # Predict
        probs = model.predict(X_batch, batch_size=64, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        # Compute accuracy
        correct = (y_pred == y_true).sum()
        acc = correct / len(y_true) if len(y_true) > 0 else 0.0

        print(f"  Overall accuracy: {acc*100:.2f}% ({correct}/{len(y_true)})")

        # Confusion breakdown
        confusion_counts = {}
        for pred_id in range(len(inferred_classes)):
            pred_class = id_to_class[pred_id]
            mask = y_pred == pred_id
            count = mask.sum()
            if count > 0:
                confusion_counts[pred_class] = count

        # Sort by count descending
        sorted_confusion = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Misclassification breakdown:")
        for pred_class, count in sorted_confusion:
            pct = (count / len(y_true)) * 100
            marker = "✓" if pred_class == target_class else "✗"
            print(f"    {marker} → {pred_class:>8}: {count:>4} samples ({pct:>5.1f}%)")

    # Generate full confusion matrix for 16QAM across all SNRs
    print(f"\n{'='*80}")
    print(f"{'FULL CONFUSION MATRIX - 16QAM (All SNRs Combined)':^80}")
    print(f"{'='*80}\n")

    all_16qam_items = []
    for snr in TARGET_SNRS:
        key = ('16QAM', snr)
        all_16qam_items.extend(by_class_snr.get(key, []))

    if all_16qam_items:
        X_batch = []
        y_true = []
        for file_path, class_id, _ in all_16qam_items:
            if not os.path.exists(file_path):
                continue
            try:
                img = tf.keras.utils.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                arr = tf.keras.utils.img_to_array(img)
                X_batch.append(arr)
                y_true.append(class_id)
            except Exception:
                continue

        X_batch = np.array(X_batch, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int64)

        probs = model.predict(X_batch, batch_size=64, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        cm = _confusion_matrix(y_true, y_pred, len(inferred_classes))

        # Extract row for 16QAM
        target_row = cm[target_class_id, :]
        total = target_row.sum()

        print(f"{'Predicted Class':<15} {'Count':>8} {'Percentage':>12}")
        print(f"{'─'*40}")
        for pred_id, count in enumerate(target_row):
            if count > 0:
                pred_class = id_to_class[pred_id]
                pct = (count / total) * 100 if total > 0 else 0.0
                marker = "✓" if pred_class == '16QAM' else "✗"
                print(f"{marker} {pred_class:<13} {count:>8} {pct:>11.1f}%")

        # Generate heatmap using imshow
        fig, ax = plt.subplots(figsize=(12, 3))

        # Use only 16QAM row for readability
        cm_16qam = cm[target_class_id:target_class_id+1, :]
        im = ax.imshow(cm_16qam, cmap='Blues', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(range(len(inferred_classes)))
        ax.set_xticklabels(inferred_classes, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['16QAM'])

        # Add text annotations
        for j in range(len(inferred_classes)):
            text = ax.text(j, 0, f'{cm_16qam[0, j]}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

        ax.set_title(f'16QAM Confusion Matrix (All SNRs, {total} samples)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=11)
        ax.set_ylabel('True Class', fontsize=11)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sample Count', fontsize=10)

        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, f'16qam_confusion_heatmap_{model_name}.png')
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved heatmap: {heatmap_path}")
        plt.close()

        # Save confusion as JSON
        confusion_data = {
            'model': model_name,
            'target_class': '16QAM',
            'total_16qam_samples': int(total),
            'confusion_by_predicted_class': {
                id_to_class[i]: {
                    'count': int(cm[target_class_id, i]),
                    'percentage': float((cm[target_class_id, i] / total) * 100) if total > 0 else 0.0
                }
                for i in range(len(inferred_classes))
            },
            'accuracy': float((cm[target_class_id, target_class_id] / total) * 100) if total > 0 else 0.0
        }

        json_path = os.path.join(output_dir, f'16qam_confusion_{model_name}.json')
        with open(json_path, 'w') as f:
            json.dump(confusion_data, f, indent=2)
        print(f"✓ Saved JSON: {json_path}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze 16QAM confusion patterns')
    parser.add_argument('--model-path', type=str,
                       default='models/squeezenet_sampler_g_20260224_155039.h5',
                       help='Path to trained model')
    parser.add_argument('--model-name', type=str,
                       default='Config_G_Adaptive',
                       help='Name for results')
    parser.add_argument('--output-dir', type=str,
                       default='results_local/phase3_clean_eval',
                       help='Directory to save results')

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"{'ANALYZING CONFUSION PATTERNS':^80}")
    print(f"Model: {args.model_name}")
    print(f"Path: {args.model_path}")
    print("="*80 + "\n")

    analyze_model_confusion(args.model_path, args.model_name, args.output_dir)
