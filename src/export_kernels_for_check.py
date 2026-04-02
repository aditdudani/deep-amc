"""
Export all kernels defined in kernel_grid_search.py for quick verification.

Generates:
1) One combined PNG montage with all kernels
2) One combined CSV containing all kernel matrices + metadata

Usage:
    python src/export_kernels_for_check.py
    python src/export_kernels_for_check.py --out-dir results/analysis/kernel_debug
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from kernel_grid_search import define_kernel_configs, calculate_shift


def save_kernel_image(kernel, title, out_path):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(kernel, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)

    h, w = kernel.shape
    for i in range(h):
        for j in range(w):
            val = int(kernel[i, j])
            text_color = "white" if val > kernel.max() * 0.45 else "black"
            plt.text(j, i, str(val), ha="center", va="center", fontsize=8, color=text_color)

    plt.xticks(range(w))
    plt.yticks(range(h))
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_all_kernels_montage(configs, out_path):
    """Save all kernels in one combined figure for quick visual cross-check."""
    ordered = list(configs.items())
    n = len(ordered)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, (config_id, cfg) in enumerate(ordered):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        kernel = cfg["kernel"]
        shift = calculate_shift(kernel)

        im = ax.imshow(kernel, cmap="viridis")
        ax.set_title(f"{config_id} | {cfg['name']}\nshift={shift}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # annotate small kernels to keep readability
        h, w = kernel.shape
        if h <= 7:
            for i in range(h):
                for j in range(w):
                    val = int(kernel[i, j])
                    text_color = "white" if val > kernel.max() * 0.45 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=text_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_all_kernels_csv(configs, out_path):
    """Save all kernels into one rectangular CSV for Rainbow CSV alignment."""
    max_size = max(cfg["size"] for cfg in configs.values())

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["config_id", "row_index"]
        header += [f"c{j}" for j in range(max_size)]
        writer.writerow(header)

        for config_id, cfg in configs.items():
            kernel = cfg["kernel"]

            # One CSV row per kernel matrix row, padded to max kernel size
            for row_index, row in enumerate(kernel.tolist()):
                padded = [int(v) for v in row] + [""] * (max_size - len(row))
                writer.writerow([
                    config_id,
                    row_index,
                    *padded,
                ])


def main():
    parser = argparse.ArgumentParser(description="Export all kernels for manual cross-check")
    parser.add_argument("--out-dir", default="results/analysis/kernel_debug", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configs = define_kernel_configs()
    csv_path = os.path.join(args.out_dir, "all_kernels.csv")
    montage_path = os.path.join(args.out_dir, "all_kernels_montage.png")

    save_all_kernels_csv(configs, csv_path)
    save_all_kernels_montage(configs, montage_path)

    print(f"Exported {len(configs)} kernels to: {args.out_dir}")
    print(f"CSV: {csv_path}")
    print(f"Montage: {montage_path}")


if __name__ == "__main__":
    main()
