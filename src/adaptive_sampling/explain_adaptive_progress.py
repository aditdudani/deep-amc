import os
import json
from typing import Dict, List, Tuple

"""
Explain adaptive progress in plain language using existing run artifacts.
Inputs: results/adaptive_sampling/<RUN_TAG>/ with weights_epoch*.json and confusion_epoch*.json
Output: prints epoch-to-epoch focus shifts and resulting accuracy shifts, and writes a summary file.
Run:
  python src/adaptive_sampling/explain_adaptive_progress.py --dir results/adaptive_sampling/<RUN_TAG>
If --dir is omitted, it will default to results/adaptive_sampling and pick the latest timestamped folder.
"""

import argparse

def _list_epochs(dir_path: str, prefix: str) -> List[int]:
    eps = []
    for name in os.listdir(dir_path):
        if name.startswith(prefix) and name.endswith('.json'):
            try:
                e = int(name[len(prefix):-5])
                eps.append(e)
            except Exception:
                continue
    return sorted(eps)

def _load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def _per_snr_avg_acc(confusion_json: dict) -> Dict[int, float]:
    cps = confusion_json.get('confusion_per_snr', {})
    out: Dict[int, float] = {}
    for snr_key, mat in cps.items():
        # snr_key may be str or int
        snr = int(snr_key)
        total = 0
        correct = 0
        for i, row in enumerate(mat):
            row_sum = int(sum(row))
            total += row_sum
            if i < len(row):
                correct += int(row[i])
        out[snr] = (correct / total) if total > 0 else 0.0
    return out

def _per_snr_share(weights_json: dict) -> Dict[int, float]:
    snrs: List[int] = [int(s) for s in weights_json.get('snrs', [])]
    W: List[List[float]] = weights_json.get('weights', [])
    if not W or not snrs:
        return {}
    import numpy as np
    Wm = np.array(W, dtype=float)
    shares: Dict[int, float] = {}
    for j, snr in enumerate(snrs):
        shares[int(snr)] = float(Wm[:, j].sum())
    return shares

def _per_bucket_shifts(weights_e: dict, weights_ep1: dict, acc_e: dict, acc_ep1: dict) -> List[Tuple[int,int,float,float]]:
    """Return top (class,snr) pairs with positive weight delta and positive acc delta.
    Output tuples: (class_id, snr, dW, dAcc)
    Uses per_class_snr_acc stored in weights json; falls back to snr-only if missing.
    """
    W_e = weights_e.get('weights', [])
    W_p = weights_ep1.get('weights', [])
    snrs: List[int] = [int(s) for s in weights_e.get('snrs', [])]
    A_e = weights_e.get('per_class_snr_acc', None)
    A_p = weights_ep1.get('per_class_snr_acc', None)
    out: List[Tuple[int,int,float,float]] = []
    if not (W_e and W_p and snrs):
        return out
    import numpy as np
    W0 = np.array(W_e, dtype=float)
    W1 = np.array(W_p, dtype=float)
    if A_e is not None and A_p is not None:
        A0 = np.array(A_e, dtype=float)
        A1 = np.array(A_p, dtype=float)
        num_c, num_s = A0.shape
        for c in range(num_c):
            for j, snr in enumerate(snrs):
                dW = float(W1[c, j] - W0[c, j])
                dA = float(A1[c, j] - A0[c, j])
                if dW > 0 and dA > 0:
                    out.append((c, int(snr), dW, dA))
    else:
        # Fallback: use snr-level accuracy deltas if class-level not present
        for j, snr in enumerate(snrs):
            dW = float(W1[:, j].sum() - W0[:, j].sum())
            dA = float(acc_ep1.get(int(snr), 0.0) - acc_e.get(int(snr), 0.0))
            if dW > 0 and dA > 0:
                out.append((-1, int(snr), dW, dA))
    # Sort by combined impact: prioritize higher accuracy gain; break ties by weight increase
    out.sort(key=lambda t: (t[3], t[2]), reverse=True)
    return out

def pick_latest_run(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Directory not found: {base_dir}")
    # Choose latest timestamp-like folder
    candidates = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not candidates:
        raise SystemExit(f"No run directories under {base_dir}")
    return sorted(candidates)[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join('results', 'adaptive_sampling'), help='Run directory or parent folder')
    ap.add_argument('--top', type=int, default=5, help='Top-N positive (focus -> accuracy) shifts to show')
    args = ap.parse_args()

    run_dir = args.dir
    if os.path.isdir(run_dir) and any(fn.startswith('weights_epoch') for fn in os.listdir(run_dir)):
        pass  # run_dir is a specific run
    else:
        run_dir = pick_latest_run(run_dir)

    weights_epochs = _list_epochs(run_dir, 'weights_epoch')
    conf_epochs = _list_epochs(run_dir, 'confusion_epoch')
    common_epochs = sorted(set(weights_epochs).intersection(conf_epochs))
    if len(common_epochs) < 2:
        print(f"Not enough epochs with both weights and confusion in {run_dir}")
        return

    print(f"Explaining adaptive progress for: {run_dir}")

    # Collect per-epoch summaries
    per_epoch_share: Dict[int, Dict[int,float]] = {}
    per_epoch_acc: Dict[int, Dict[int,float]] = {}
    weights_jsons: Dict[int, dict] = {}

    for e in common_epochs:
        wj = _load_json(os.path.join(run_dir, f'weights_epoch{e}.json'))
        cj = _load_json(os.path.join(run_dir, f'confusion_epoch{e}.json'))
        weights_jsons[e] = wj
        per_epoch_share[e] = _per_snr_share(wj)
        per_epoch_acc[e] = _per_snr_avg_acc(cj)

    # Epoch-to-epoch explanations
    lines: List[str] = []
    lines.append("\n=== Focus Shifts and Accuracy Response ===")
    for e in common_epochs[:-1]:
        ep1 = e + 1
        if ep1 not in per_epoch_share or ep1 not in per_epoch_acc:
            continue
        share_e = per_epoch_share[e]
        share_p = per_epoch_share[ep1]
        acc_e = per_epoch_acc[e]
        acc_p = per_epoch_acc[ep1]
        # Compute deltas per SNR
        snrs = sorted(set(share_e.keys()).union(share_p.keys()))
        deltas = []
        for snr in snrs:
            dW = share_p.get(snr, 0.0) - share_e.get(snr, 0.0)
            dA = acc_p.get(snr, 0.0) - acc_e.get(snr, 0.0)
            deltas.append((snr, dW, dA))
        # Sort SNRs by weight increase
        deltas.sort(key=lambda t: t[1], reverse=True)
        lines.append(f"\nEpoch {e} -> {ep1}")
        for snr, dW, dA in deltas:
            arrow_w = "+" if dW > 0 else ("-" if dW < 0 else "=")
            arrow_a = "+" if dA > 0 else ("-" if dA < 0 else "=")
            lines.append(f"  SNR {snr:>2}: weight {arrow_w}{abs(dW):.4f} -> accuracy {arrow_a}{abs(dA):.4f}")
        # Top class,SNR pairs where weight↑ and accuracy↑
        top_pairs = _per_bucket_shifts(weights_jsons[e], weights_jsons[ep1], acc_e, acc_p)[:args.top]
        if top_pairs:
            lines.append("  Top (class,SNR) where focus coincided with accuracy gain:")
            for c, snr, dW, dA in top_pairs:
                c_disp = f"Class {c}" if c >= 0 else "(class-agnostic)"
                lines.append(f"    {c_disp} @ SNR {snr:>2}: Δweight=+{dW:.4f}, Δacc=+{dA:.4f}")
        else:
            lines.append("  No clear (class,SNR) pairs with both weight↑ and accuracy↑ this step.")

    # Write summary file and print
    out_path = os.path.join(run_dir, 'explain_summary.txt')
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved explanation to: {out_path}")

if __name__ == '__main__':
    main()
