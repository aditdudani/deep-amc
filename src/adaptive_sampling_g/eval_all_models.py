#!/usr/bin/env python3
"""
Run clean evaluation on all three models and generate comparison report.
"""

import os
import sys
import json
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval_validation_clean import eval_validation_clean


def run_all_evaluations():
    """Evaluate all three models and save comparison."""

    models = [
        {
            'model_path': 'results_local/phase2_matrix/model_config_A.keras',
            'model_name': 'Config_A_Baseline',
            'description': 'Phase 2 Winner - K02 3x3 Cross'
        },
        {
            'model_path': 'results_local/phase2_matrix/model_config_G.keras',
            'model_name': 'Config_G_Baseline',
            'description': 'Phase 2 Runner-up - K20 3x3 Cross Centered'
        },
        {
            'model_path': 'models/squeezenet_sampler_g_20260224_155039.h5',
            'model_name': 'Config_G_Adaptive',
            'description': 'Adaptive Sampling on Config G'
        },
    ]

    output_dir = 'results_local/phase3_clean_eval'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 3 CLEAN EVALUATION - ALL MODELS")
    print("=" * 80)

    results = []
    for model_info in models:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_info['description']}")
        print(f"Model: {model_info['model_path']}")
        print(f"{'='*80}\n")

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
            print(f"[ERROR] Failed to evaluate {model_info['model_name']}: {e}")
            continue

    # Generate comparison report
    print(f"\n{'='*80}")
    print("COMPARISON REPORT")
    print(f"{'='*80}\n")

    # Overall accuracy comparison
    print("Overall Accuracy Comparison:")
    print("-" * 60)
    for r in results:
        acc = r['result']['overall_accuracy'] * 100
        count = r['result']['overall_count']
        print(f"  {r['model_name']:25} -> {acc:6.2f}% (n={count})")

    # Per-SNR comparison
    print(f"\nPer-SNR Accuracy Comparison:")
    print("-" * 60)
    snrs = results[0]['result']['snrs']
    for snr in snrs:
        print(f"\n  SNR = {snr} dB:")
        for r in results:
            acc = r['result']['accuracy_by_snr'][snr] * 100
            count = r['result']['counts_by_snr'][snr]
            print(f"    {r['model_name']:25} -> {acc:6.2f}% (n={count})")

    # Per-class comparison
    print(f"\nPer-Class Overall Accuracy Comparison:")
    print("-" * 60)
    classes = results[0]['result']['classes']
    for cls in classes:
        print(f"\n  {cls}:")
        for r in results:
            if cls in r['result']['per_class_overall_accuracy']:
                acc = r['result']['per_class_overall_accuracy'][cls] * 100
                print(f"    {r['model_name']:25} -> {acc:6.2f}%")

    # Save detailed comparison JSON
    comparison = {
        'timestamp': str(os.popen('date -u +%Y-%m-%dT%H:%M:%SZ').read().strip()),
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
    print(f"\n[save] Comparison JSON: {comparison_path}")

    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_all_evaluations()
