"""
Generate all graphs for kernel search + phase2 matrix CSV results.

Usage:
    python src/generate_all_result_graphs.py
    python src/generate_all_result_graphs.py \
        --kernel-csv results/analysis/kernel_search/kernel_rankings_20260218_011931.csv \
        --phase2-csv results/analysis/phase2_matrix/phase2_rankings_20260218_095516.csv \
        --out-dir results/analysis/all_graphs
"""

import os
import argparse
import csv
import matplotlib.pyplot as plt


DEFAULT_KERNEL_CSV = "results/analysis/kernel_search/kernel_rankings_20260218_011931.csv"
DEFAULT_PHASE2_CSV = "results/analysis/phase2_matrix/phase2_rankings_20260218_095516.csv"
DEFAULT_OUT_DIR = "results/analysis/all_graphs"
SNR_COLUMNS = ["snr_0dB", "snr_2dB", "snr_4dB", "snr_6dB", "snr_8dB", "snr_10dB"]


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_bool(value):
    return str(value).strip().lower() == "true"


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_kernel_overall_bar(rows, out_dir):
    data = sorted(rows, key=lambda r: _to_float(r.get("best_acc", 0.0)), reverse=True)
    plt.figure(figsize=(12, 6))
    labels = [r["config_id"] for r in data]
    vals = [_to_float(r["best_acc"]) for r in data]
    plt.bar(labels, vals, color="steelblue")
    plt.ylabel("Best Accuracy (%)")
    plt.xlabel("Kernel Config")
    plt.title("Kernel Search: Best Accuracy by Config")
    plt.xticks(rotation=45)
    save_fig(os.path.join(out_dir, "kernel_overall_accuracy_bar.png"))


def plot_kernel_cost_vs_accuracy(rows, out_dir):
    plt.figure(figsize=(8, 6))
    for row in rows:
        cost = _to_float(row.get("cost_proxy", 0.0))
        acc = _to_float(row.get("best_acc", 0.0))
        color = "tab:red" if _to_bool(row.get("uses_dsp", "False")) else "tab:green"
        plt.scatter(cost, acc, s=80, color=color)
        plt.text(cost, acc, f" {row['config_id']}", fontsize=8)

    plt.xlabel("Hardware Cost Proxy")
    plt.ylabel("Best Accuracy (%)")
    plt.title("Kernel Search: Cost vs Accuracy")
    plt.grid(alpha=0.3)
    save_fig(os.path.join(out_dir, "kernel_cost_vs_accuracy.png"))


def plot_kernel_per_snr_lines(rows, out_dir):
    plt.figure(figsize=(12, 7))
    x_labels = ["0", "2", "4", "6", "8", "10"]

    data = sorted(rows, key=lambda r: _to_float(r.get("best_acc", 0.0)), reverse=True)
    for row in data:
        y = [_to_float(row.get(col, 0.0)) for col in SNR_COLUMNS]
        plt.plot(x_labels, y, marker="o", linewidth=1.5, label=row["config_id"])

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Kernel Search: Per-SNR Accuracy Curves")
    plt.grid(alpha=0.3)
    plt.legend(ncol=4, fontsize=8)
    save_fig(os.path.join(out_dir, "kernel_per_snr_lines.png"))


def plot_phase2_overall_bar(rows, out_dir):
    data = sorted(rows, key=lambda r: _to_float(r.get("best_acc", 0.0)), reverse=True)
    plt.figure(figsize=(10, 6))
    colors = [{"Single": "tab:blue", "Dual": "tab:orange", "Triple": "tab:purple"}.get(r.get("tier", ""), "tab:gray") for r in data]
    labels = [f"{r['config_id']} ({r['name']})" for r in data]
    vals = [_to_float(r["best_acc"]) for r in data]

    plt.bar(labels, vals, color=colors)
    plt.ylabel("Best Accuracy (%)")
    plt.xlabel("Phase 2 Config")
    plt.title("Phase 2: Best Accuracy by Architecture")
    plt.xticks(rotation=30, ha="right")
    save_fig(os.path.join(out_dir, "phase2_overall_accuracy_bar.png"))


def plot_phase2_cost_vs_accuracy(rows, out_dir):
    plt.figure(figsize=(8, 6))
    color_map = {"Single": "tab:blue", "Dual": "tab:orange", "Triple": "tab:purple"}

    for row in rows:
        cost = _to_float(row.get("cost", 0.0))
        acc = _to_float(row.get("best_acc", 0.0))
        c = color_map.get(row.get("tier", ""), "tab:gray")
        plt.scatter(cost, acc, s=100, color=c)
        plt.text(cost, acc, f" {row['config_id']}", fontsize=9)

    plt.xlabel("Cost")
    plt.ylabel("Best Accuracy (%)")
    plt.title("Phase 2: Cost vs Accuracy")
    plt.grid(alpha=0.3)
    save_fig(os.path.join(out_dir, "phase2_cost_vs_accuracy.png"))


def plot_phase2_per_snr_lines(rows, out_dir):
    plt.figure(figsize=(10, 6))
    x_labels = ["0", "2", "4", "6", "8", "10"]

    data = sorted(rows, key=lambda r: _to_float(r.get("best_acc", 0.0)), reverse=True)
    for row in data:
        y = [_to_float(row.get(col, 0.0)) for col in SNR_COLUMNS]
        plt.plot(x_labels, y, marker="o", linewidth=2, label=row["config_id"])

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Phase 2: Per-SNR Accuracy Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig(os.path.join(out_dir, "phase2_per_snr_lines.png"))


def plot_cross_file_top_comparison(kernel_rows, phase2_rows, out_dir):
    kernel_top = sorted(kernel_rows, key=lambda r: _to_float(r.get("best_acc", 0.0)), reverse=True)[:3]
    phase2_top = sorted(phase2_rows, key=lambda r: _to_float(r.get("best_acc", 0.0)), reverse=True)[:3]

    labels = [r["config_id"] + " [Kernel]" for r in kernel_top] + [r["config_id"] + " [Phase2]" for r in phase2_top]
    vals = [_to_float(r["best_acc"]) for r in kernel_top] + [_to_float(r["best_acc"]) for r in phase2_top]
    colors = ["tab:green"] * 3 + ["tab:purple"] * 3

    plt.figure(figsize=(11, 6))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Best Accuracy (%)")
    plt.xlabel("Top Configurations")
    plt.title("Top-3 Kernel Search vs Top-3 Phase 2")
    plt.xticks(rotation=20, ha="right")
    save_fig(os.path.join(out_dir, "cross_top3_comparison.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate all graphs for kernel + phase2 CSV files")
    parser.add_argument("--kernel-csv", default=DEFAULT_KERNEL_CSV, help="Kernel search CSV path")
    parser.add_argument("--phase2-csv", default=DEFAULT_PHASE2_CSV, help="Phase 2 matrix CSV path")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for figures")
    args = parser.parse_args()

    ensure_out_dir(args.out_dir)

    kernel_rows = read_csv_rows(args.kernel_csv)
    phase2_rows = read_csv_rows(args.phase2_csv)

    required_kernel = {"config_id", "cost_proxy", "uses_dsp", "best_acc", *SNR_COLUMNS}
    required_phase2 = {"config_id", "name", "tier", "cost", "best_acc", *SNR_COLUMNS}

    if not kernel_rows:
        raise ValueError("Kernel CSV has no rows")
    if not phase2_rows:
        raise ValueError("Phase2 CSV has no rows")

    if not required_kernel.issubset(set(kernel_rows[0].keys())):
        missing = required_kernel - set(kernel_rows[0].keys())
        raise ValueError(f"Kernel CSV missing columns: {sorted(missing)}")

    if not required_phase2.issubset(set(phase2_rows[0].keys())):
        missing = required_phase2 - set(phase2_rows[0].keys())
        raise ValueError(f"Phase2 CSV missing columns: {sorted(missing)}")

    plot_kernel_overall_bar(kernel_rows, args.out_dir)
    plot_kernel_cost_vs_accuracy(kernel_rows, args.out_dir)
    plot_kernel_per_snr_lines(kernel_rows, args.out_dir)

    plot_phase2_overall_bar(phase2_rows, args.out_dir)
    plot_phase2_cost_vs_accuracy(phase2_rows, args.out_dir)
    plot_phase2_per_snr_lines(phase2_rows, args.out_dir)

    plot_cross_file_top_comparison(kernel_rows, phase2_rows, args.out_dir)

    print("\nDone. Generated 7 plots in:")
    print(args.out_dir)


if __name__ == "__main__":
    main()
