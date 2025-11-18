"""Inspect evolution of adaptive sampler weights and (optionally) confusion metrics.
    # SNR accuracy trend across epochs
    python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling/<RUN_TAG> --snr-trend --with-confusion

Examples:
    # All epochs, top 12 buckets, entropy metrics
    python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling/<RUN_TAG> --top 12 --entropy

    # Latest only with confusion summary (if confusion_epoch*.json present)
    python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling/<RUN_TAG> --latest --with-confusion

    # Concentration warnings customized
    python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling/<RUN_TAG> --warn-max-mult 6 --warn-entropy-ratio 0.50

Metrics:
    - Entropy H and ratio H/H_uniform (uniform entropy = log(C*S)).
    - KL divergence KL(p||u) = log(C*S) - H.
    - Gini coefficient (0 uniform, →1 concentrated).
    - Max bucket multiple vs uniform baseline.
    - Per-SNR and per-class total shares.
    - Optional confusion-derived per-SNR diagonal accuracy (avg, min, max).

Heuristics:
    - H/H_uniform < 0.6: concentrated; < 0.4: sharply peaked.
    - Max bucket > 6× uniform early may risk over-focus.
    - Gini > 0.6 indicates strong skew; monitor tail performance.
"""
import os
import json
import math
import argparse
from typing import List, Tuple, Dict

def _load_weight_files(directory: str) -> List[Tuple[int, str]]:
    files = []
    for name in os.listdir(directory):
        if name.startswith('weights_epoch') and name.endswith('.json'):
            try:
                epoch = int(name[len('weights_epoch'):-len('.json')])
            except ValueError:
                continue
            files.append((epoch, os.path.join(directory, name)))
    return sorted(files)

def _format_bucket(class_idx: int, snr: int) -> str:
    return f"C{class_idx}_S{snr}"


def _load_confusion_files(run_dir: str) -> List[Tuple[int, str]]:
    files = []
    for name in os.listdir(run_dir):
        if name.startswith('confusion_epoch') and name.endswith('.json'):
            try:
                epoch = int(name[len('confusion_epoch'):-len('.json')])
            except ValueError:
                continue
            files.append((epoch, os.path.join(run_dir, name)))
    return sorted(files)


def _summarize_confusion_per_snr(conf_path: str) -> Dict[int, Dict[str, float]]:
    with open(conf_path, 'r') as f:
        data = json.load(f)
    cps = data.get('confusion_per_snr', {})
    summary = {}
    for snr_str, matrix in cps.items():
        # matrix: list of rows; each row list of counts predicted per class
        row_acc = []
        total_correct = 0
        total_all = 0
        for row_idx, row in enumerate(matrix):
            row_sum = sum(row)
            if row_sum > 0:
                correct = row[row_idx]
                acc = correct / row_sum
                row_acc.append(acc)
                total_correct += correct
                total_all += row_sum
        if total_all > 0:
            avg_acc = total_correct / total_all
        else:
            avg_acc = 0.0
        summary[int(snr_str)] = {
            'avg_acc': avg_acc,
            'min_acc': min(row_acc) if row_acc else 0.0,
            'max_acc': max(row_acc) if row_acc else 0.0,
        }
    return summary


def _entropy_metrics(flat_probs: List[float]) -> Tuple[float, float, float, float]:
    # Returns (H, H_uniform, KL, gini)
    import math
    n = len(flat_probs)
    H = -sum(p * math.log(p + 1e-12) for p in flat_probs)
    H_uniform = math.log(n) if n > 0 else 0.0
    KL = H_uniform - H
    # Gini coefficient: 1 - sum_i p_i^2 / (sum_i p_i)^2 for probs already summing to 1 => 1 - sum p_i^2
    gini = 1.0 - sum(p * p for p in flat_probs)
    return H, H_uniform, KL, gini

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/adaptive_sampling', help='Directory containing weights_epoch*.json files or a specific run directory')
    ap.add_argument('--top', type=int, default=8, help='Show top-N buckets by weight')
    ap.add_argument('--entropy', action='store_true', help='Report entropy and KL divergence vs uniform per epoch')
    ap.add_argument('--latest', action='store_true', help='Show only latest epoch metrics')
    ap.add_argument('--with-confusion', action='store_true', help='Include confusion-derived per-SNR accuracy if files present')
    ap.add_argument('--warn-max-mult', type=float, default=6.0, help='Warn if max bucket exceeds this multiple of uniform')
    ap.add_argument('--warn-entropy-ratio', type=float, default=0.40, help='Warn if H/H_uniform falls below this value')
    ap.add_argument('--show-per-class', action='store_true', help='Show per-class total weight share')
    ap.add_argument('--snr-trend', action='store_true', help='After epoch summaries, print per-SNR avg accuracy trend (requires confusion files & not --latest)')
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Directory not found: {args.dir}")

    weight_files = _load_weight_files(args.dir)
    if not weight_files:
        raise SystemExit("No weights_epoch*.json files found yet.")

    print(f"Found {len(weight_files)} epochs of weights in {args.dir}\n")

    target_files = [weight_files[-1]] if args.latest else weight_files
    confusion_files = _load_confusion_files(args.dir) if args.with_confusion else []
    confusion_map = {ep: p for ep, p in confusion_files}

    snr_trend: Dict[int, List[Tuple[int, float]]] = {}
    for epoch, path in target_files:
        with open(path, 'r') as f:
            data = json.load(f)
        weights = data['weights']  # 2D list [num_classes][num_snrs]
        snrs = data.get('snrs', [])
        num_classes = len(weights)
        num_snrs = len(weights[0]) if weights else 0
        flat = []
        for c in range(num_classes):
            for j in range(num_snrs):
                flat.append((weights[c][j], c, snrs[j] if j < len(snrs) else j))
        flat.sort(reverse=True)
        total = sum(w for w,_,_ in flat) or 1.0
        uniform = 1.0 / len(flat)
        probs = [w/total for w,_,_ in flat]
        H, H_uniform, KL, gini = _entropy_metrics(probs) if args.entropy else (None, None, None, None)
        max_bucket = flat[0][0] if flat else 0.0
        max_mult = max_bucket / uniform if uniform > 0 else 0.0
        header = f"Epoch {epoch} (sum={total:.4f}, uniform~{uniform:.5f})"
        if args.entropy:
            header += f" H={H:.4f} H/Hu={H/H_uniform:.3f} KL={KL:.4f} Gini={gini:.3f}"
        header += f" max_mult={max_mult:.2f}"
        warnings = []
        if max_mult > args.warn_max_mult:
            warnings.append(f"max_mult>{args.warn_max_mult}")
        if args.entropy and (H/H_uniform) < args.warn_entropy_ratio:
            warnings.append(f"entropy_ratio<{args.warn_entropy_ratio}")
        if warnings:
            header += " WARN:[" + ",".join(warnings) + "]"
        print(header)
        print(" Top buckets:")
        for w, c, snr in flat[:args.top]:
            ratio = w / uniform if uniform > 0 else 0
            print(f"  {_format_bucket(c, snr):>10}  w={w:.5f}  x_uniform={ratio:5.2f}")
        # Aggregate per SNR
        per_snr = {}
        for w, c, snr in flat:
            per_snr[snr] = per_snr.get(snr, 0.0) + w
        print(" Per-SNR weight share:")
        for snr in sorted(per_snr):
            print(f"  SNR {snr:>2}: {per_snr[snr]:.4f}")
        if args.show_per_class:
            per_class = [sum(weights[c]) for c in range(num_classes)]
            print(" Per-class weight share:")
            for c, val in enumerate(per_class):
                print(f"  Class {c}: {val:.4f}")
        if args.with_confusion and epoch in confusion_map:
            conf_summary = _summarize_confusion_per_snr(confusion_map[epoch])
            print(" Per-SNR avg/min/max diagonal accuracy:")
            for snr in sorted(conf_summary):
                cs = conf_summary[snr]
                print(f"  SNR {snr:>2}: avg={cs['avg_acc']:.4f} min={cs['min_acc']:.4f} max={cs['max_acc']:.4f}")
                if args.snr_trend and not args.latest:
                    snr_trend.setdefault(snr, []).append((epoch, cs['avg_acc']))
        print()

    if args.snr_trend and not args.latest and snr_trend:
        print("=== Per-SNR Average Accuracy Trend ===")
        # Sort epochs
        all_epochs = sorted({e for lst in snr_trend.values() for e,_ in lst})
        header = "Epoch" + "".join([f"\tSNR{snr}" for snr in sorted(snr_trend)])
        print(header)
        for ep in all_epochs:
            row = [str(ep)]
            for snr in sorted(snr_trend):
                vals = {e:acc for e,acc in snr_trend[snr]}
                row.append(f"{vals.get(ep,'-')}")
            print("\t".join(row))
        print("\nInterpretation: rising low-SNR columns indicate successful adaptive emphasis; stagnation while weights concentrate suggests lowering beta or adding replay.")

if __name__ == '__main__':
    main()
