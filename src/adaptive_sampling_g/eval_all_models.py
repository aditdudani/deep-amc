#!/usr/bin/env python3
"""
Run clean evaluation on all three models and generate comparison report.
"""

import os
import sys
import json
import warnings

# Suppress TensorFlow logging BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

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

    print(f"\n╔" + "═"*78 + "╗")
    print(f"║{f'Results saved to: {output_dir}':^78}║")
    print(f"╚" + "═"*78 + "╝\n")


if __name__ == '__main__':
    run_all_evaluations()
