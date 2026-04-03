#!/usr/bin/env python3
"""
Run clean evaluation on all three models and generate comparison report + visualizations.
Auto-detects latest adaptive sampling model.
"""

import os
import sys
import json
import warnings
import glob

# Suppress TensorFlow logging BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval_validation_clean import eval_validation_clean


def find_latest_adaptive_model():
    """Find the most recent adaptive sampling model in results/adaptive_sampling_g/ or models/ (fallback)"""
    # Try new results/ structure first
    pattern = 'results/adaptive_sampling_g/*/model.h5'
    models = glob.glob(pattern)

    # Fallback to old models/ directory for backward compatibility
    if not models:
        pattern = 'models/squeezenet_sampler_g_*.h5'
        models = glob.glob(pattern)

    if not models:
        raise FileNotFoundError(f"No adaptive models found in results/adaptive_sampling_g/ or models/")

    # Sort by modification time, return most recent
    latest = max(models, key=os.path.getmtime)
    return latest


def run_all_evaluations():
    """Evaluate all three models and save comparison."""

    # Auto-detect latest adaptive model
    latest_adaptive = find_latest_adaptive_model()

    models = [
        {
            'model_path': 'results/baselines/config_a/model.keras',
            'model_name': 'Config_A_Baseline',
            'description': 'Phase 2 Winner - K02 3x3 Cross'
        },
        {
            'model_path': 'results/baselines/config_g/model.keras',
            'model_name': 'Config_G_Baseline',
            'description': 'Phase 2 Runner-up - K20 3x3 Cross Centered'
        },
        {
            'model_path': latest_adaptive,
            'model_name': 'Config_G_Adaptive',
            'description': f'Adaptive Sampling on Config G ({os.path.basename(os.path.dirname(latest_adaptive))})'
        },
    ]

    output_dir = 'results/comparisons/phase3_all_models'
    os.makedirs(output_dir, exist_ok=True)

    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + f"{'PHASE 3 CLEAN EVALUATION - ALL MODELS':^78}" + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")

    results = []
    for i, model_info in enumerate(models, 1):
        print(f"\n[{i}/3] Evaluating: {model_info['description']}")
        print(f"      Model: {model_info['model_path']}")
        print(f"      {'─'*74}")

        try:
            result = eval_validation_clean(
                model_path=model_info['model_path'],
                metadata_val_csv='data/processed_g/metadata_val.csv',
                output_dir=output_dir,
                model_name=model_info['model_name'],
                batch_size=64
            )
            results.append({
                'model_name': model_info['model_name'],
                'description': model_info['description'],
                'result': result
            })
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            continue

    # Generate comparison report
    print(f"\n{'╔' + '═'*78 + '╗'}")
    print(f"║{' COMPARISON REPORT ':^78}║")
    print(f"{'╚' + '═'*78 + '╝'}")

    # Overall accuracy comparison
    print(f"\n{'Overall Accuracy Comparison:':}")
    print(f"{'─'*78}")
    for r in results:
        acc = r['result']['overall_accuracy'] * 100
        count = r['result']['overall_count']
        delta_a = None
        if r['model_name'] != 'Config_A_Baseline':
            baseline_acc = [x['result']['overall_accuracy'] for x in results if x['model_name'] == 'Config_A_Baseline'][0] * 100
            delta_a = acc - baseline_acc
            delta_str = f"  ({delta_a:+.2f}% vs Config A)"
        else:
            delta_str = ""
        print(f"  {r['model_name']:25} → {acc:6.2f}% (n={count:,}){delta_str}")

    # Per-SNR comparison
    print(f"\n{'Per-SNR Accuracy Comparison:':}")
    print(f"{'─'*78}")
    snrs = results[0]['result']['snrs']
    for snr in snrs:
        print(f"\n  SNR = {snr} dB:")
        for r in results:
            acc = r['result']['accuracy_by_snr'][snr] * 100
            count = r['result']['counts_by_snr'][snr]
            print(f"    {r['model_name']:25} → {acc:6.2f}% (n={count:,})")

    # Per-class comparison
    print(f"\n{'Per-Class Overall Accuracy Comparison:':}")
    print(f"{'─'*78}")
    classes = results[0]['result']['classes']
    for cls in classes:
        print(f"\n  {cls}:")
        for r in results:
            if cls in r['result']['per_class_overall_accuracy']:
                acc = r['result']['per_class_overall_accuracy'][cls] * 100
                print(f"    {r['model_name']:25} → {acc:6.2f}%")

    # Save detailed comparison JSON
    comparison = {
        'timestamp': os.popen('date -u +%Y-%m-%dT%H:%M:%SZ').read().strip(),
        'models_evaluated': [r['description'] for r in results],
        'results': [r['result'] for r in results],
        'summary': {
            'overall_accuracy': {r['model_name']: r['result']['overall_accuracy'] for r in results},
            'per_class_overall_accuracy': {
                r['model_name']: r['result']['per_class_overall_accuracy'] for r in results
            }
        }
    }

    comparison_path = os.path.join(output_dir, 'phase3_comparison_all_models.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Saved comparison JSON: {comparison_path}")

    # ═════════════════════════════════════════════════════════════════════════════════
    # GENERATE COMPARISON VISUALIZATIONS
    # ═════════════════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*78}")
    print(f"Generating comparison visualizations...")
    print(f"{'─'*78}")

    # 1. Overall accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [r['model_name'] for r in results]
    overall_accs = [r['result']['overall_accuracy'] * 100 for r in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(model_names, overall_accs, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(70, 82)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, acc in zip(bars, overall_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_overall_accuracy.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Overall accuracy bar chart")
    plt.close()

    # 2. Per-SNR line plot (all models on same graph)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for (r, color, marker) in zip(results, colors, markers):
        snr_vals = sorted(r['result']['accuracy_by_snr'].keys())
        accs = [r['result']['accuracy_by_snr'][s] * 100 for s in snr_vals]
        ax.plot(snr_vals, accs, marker=marker, linewidth=2.5, markersize=8,
                label=r['model_name'], color=color)

    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by SNR - All Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xticks(snrs)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_per_snr.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Per-SNR line plot")
    plt.close()

    # 3. Per-class comparison (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    classes = results[0]['result']['classes']
    x = np.arange(len(classes))
    width = 0.25

    for i, (r, color) in enumerate(zip(results, colors)):
        accs = [r['result']['per_class_overall_accuracy'][c] * 100 for c in classes]
        ax.bar(x + i*width, accs, width, label=r['model_name'], color=color, edgecolor='black', linewidth=1)

    ax.set_ylabel('Overall Accuracy per Class (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Overall Accuracy - All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_per_class.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Per-class grouped bar chart")
    plt.close()

    # 4. Combined heatmaps (all 3 models side-by-side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    snr_count = len(results[0]['result']['snrs'])
    class_count = len(classes)

    for idx, (r, ax) in enumerate(zip(results, axes)):
        per_class_snr = np.array(r['result']['per_class_snr_accuracy'])
        im = ax.imshow(per_class_snr * 100, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(range(len(snrs)))
        ax.set_xticklabels([f'{s}dB' for s in snrs], fontsize=9)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel('SNR', fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Class', fontsize=10, fontweight='bold')
        ax.set_title(r['model_name'], fontsize=11, fontweight='bold')

        # Add text annotations (smaller font for 3-column layout)
        for i in range(len(classes)):
            for j in range(len(snrs)):
                ax.text(j, i, f'{per_class_snr[i, j]*100:.0f}',
                       ha="center", va="center", color="black", fontsize=7)

    # Single colorbar for all heatmaps
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.01, aspect=40)
    cbar.set_label('Accuracy (%)', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_heatmaps_all_models.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Combined heatmaps (per-class per-SNR)")
    plt.close()

    # 5. Delta comparison vs Config A baseline
    if len(results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        config_a_acc = results[0]['result']['per_class_overall_accuracy']

        # Per-class delta
        ax = axes[0]
        for i in range(1, len(results)):
            r = results[i]
            deltas = [
                (r['result']['per_class_overall_accuracy'][c] - config_a_acc[c]) * 100
                for c in classes
            ]
            ax.plot(classes, deltas, marker='o', linewidth=2.5, markersize=8,
                   label=r['model_name'], color=colors[i])

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Accuracy Delta (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Improvement vs Config A', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        ax.set_xticklabels(classes, rotation=45, ha='right')

        # Per-SNR delta
        ax = axes[1]
        config_a_snr = results[0]['result']['accuracy_by_snr']

        for i in range(1, len(results)):
            r = results[i]
            snr_vals = sorted(r['result']['accuracy_by_snr'].keys())
            deltas = [
                (r['result']['accuracy_by_snr'][s] - config_a_snr[s]) * 100
                for s in snr_vals
            ]
            ax.plot(snr_vals, deltas, marker=markers[i], linewidth=2.5, markersize=8,
                   label=r['model_name'], color=colors[i])

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Delta (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-SNR Improvement vs Config A', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        ax.set_xticks(snr_vals)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'comparison_delta_vs_baseline.png'), dpi=150, bbox_inches='tight')
        print(f"  ✓ Delta comparison vs Config A baseline")
        plt.close()

    print(f"\n╔" + "═"*78 + "╗")
    print(f"║{f'All results saved to: {output_dir}':^78}║")
    print(f"╚" + "═"*78 + "╝\n")


if __name__ == '__main__':
    run_all_evaluations()
