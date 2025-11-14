"""Inspect evolution of adaptive sampler weights across epochs.

Run after some training epochs:
  python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling

Outputs summary tables showing top-weighted (class,SNR) buckets per epoch
and deviation from uniform distribution.
"""
import os
import json
import argparse
from typing import List, Tuple

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/adaptive_sampling', help='Directory containing weights_epoch*.json files')
    ap.add_argument('--top', type=int, default=8, help='Show top-N buckets by weight')
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Directory not found: {args.dir}")

    weight_files = _load_weight_files(args.dir)
    if not weight_files:
        raise SystemExit("No weights_epoch*.json files found yet.")

    print(f"Found {len(weight_files)} epochs of weights in {args.dir}\n")

    for epoch, path in weight_files:
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
        print(f"Epoch {epoch} (sum={total:.4f}, uniform~{uniform:.5f})")
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
        print()

if __name__ == '__main__':
    main()
