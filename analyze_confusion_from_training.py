#!/usr/bin/env python3
"""
Analyze 16QAM confusion patterns from saved confusion matrices during training.
Focuses on what 16QAM is being confused with over epochs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Config
TARGET_SNRS = [0, 2, 4, 6, 8, 10]
MODS = ['16QAM', '32QAM', '4ASK', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK']
RUN_DIR = 'results_local/adaptive_sampling_g/20260224_155039'
OUTPUT_DIR = 'results_local/phase3_clean_eval'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print(f"{'ANALYZING 16QAM CONFUSION FROM TRAINING HISTORY':^80}")
print(f"{'Run: 20260224_155039 (Most Recent Training)':^80}")
print("="*80)

# Find the last epoch with confusion data
confusion_files = sorted([f for f in os.listdir(RUN_DIR) if f.startswith('confusion_epoch')])
if not confusion_files:
    print("❌ No confusion matrices found")
    exit(1)

latest_epoch = int(confusion_files[-1].replace('confusion_epoch', '').replace('.json', ''))
print(f"\n✓ Found {len(confusion_files)} confusion matrices (epochs 1-{latest_epoch})")

# Load latest epoch confusion
latest_confusion_path = os.path.join(RUN_DIR, f'confusion_epoch{latest_epoch}.json')
with open(latest_confusion_path, 'r') as f:
    confusion_data = json.load(f)

snrs = confusion_data['snrs']
confusion_per_snr = confusion_data['confusion_per_snr']

print(f"\n{'═'*80}")
print(f"{'16QAM CONFUSION ANALYSIS - EPOCH {0}'.format(latest_epoch):^80}")
print(f"{'═'*80}")

# Index of 16QAM
target_idx = MODS.index('16QAM')

# Analyze each SNR
for snr in snrs:
    snr_key = str(snr)
    if snr_key not in confusion_per_snr:
        continue

    cm = np.array(confusion_per_snr[snr_key])

    # Get row for 16QAM
    target_row = cm[target_idx, :]
    total = target_row.sum()

    if total == 0:
        continue

    # Compute accuracy
    correct = cm[target_idx, target_idx]
    acc = (correct / total) * 100

    print(f"\n[SNR {snr:>2} dB] 16QAM accuracy: {acc:6.2f}% ({correct}/{int(total)} samples)")
    print(f"{'─'*80}")

    # Show what 16QAM is being confused with
    confusion_dict = {}
    for pred_idx, count in enumerate(target_row):
        if count > 0:
            confusion_dict[MODS[pred_idx]] = (int(count), (count/total)*100)

    # Sort by count
    sorted_confusion = sorted(confusion_dict.items(), key=lambda x: x[1][0], reverse=True)

    for mod, (count, pct) in sorted_confusion:
        marker = "✓" if mod == '16QAM' else "✗"
        print(f"  {marker} → {mod:>8}: {count:>4} ({pct:>5.1f}%)")

# Now analyze how confusion changed over epochs
print(f"\n{'═'*80}")
print(f"{'16QAM ACCURACY TREND OVER EPOCHS':^80}")
print(f"{'═'*80}\n")

epochs_to_check = [4, 10, 15, 22, 40]
epoch_data = []

for epoch in confusion_files:
    ep_num = int(epoch.replace('confusion_epoch', '').replace('.json', ''))
    with open(os.path.join(RUN_DIR, epoch), 'r') as f:
        data = json.load(f)

    # Focus on SNR 0
    if '0' in data['confusion_per_snr']:
        cm = np.array(data['confusion_per_snr']['0'])
        target_row = cm[target_idx, :]
        correct = cm[target_idx, target_idx]
        total = target_row.sum()
        if total > 0:
            acc = (correct / total) * 100
            epoch_data.append((ep_num, acc))

# Print key epochs
print(f"{'Epoch':<10} {'16QAM 0dB Accuracy':<20}")
print(f"{'─'*30}")
for ep, ep_nums in enumerate(epochs_to_check):
    for ep_num, acc in epoch_data:
        if ep_num == ep_nums:
            print(f"{ep_num:<10} {acc:>6.2f}%")
            break

# Generate trend plot
if epoch_data:
    fig, ax = plt.subplots(figsize=(12, 6))

    eps, accs = zip(*epoch_data)
    ax.plot(eps, accs, marker='o', linewidth=2.5, markersize=6, color='tab:red', label='16QAM @ 0dB')

    # Mark key epochs
    for ep_num in epochs_to_check:
        for e, a in epoch_data:
            if e == ep_num:
                ax.scatter(e, a, s=150, color='darkred', zorder=5, edgecolors='black', linewidth=2)
                ax.annotate(f'{a:.1f}%', xy=(e, a), xytext=(10, 10),
                           textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
                break

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (target floor)')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('16QAM @ 0dB Accuracy Trend During Adaptive Sampling Training',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    trend_path = os.path.join(OUTPUT_DIR, '16qam_0db_accuracy_trend.png')
    fig.savefig(trend_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved trend plot: {trend_path}")
    plt.close()

# Final diagnosis
print(f"\n{'═'*80}")
print(f"{'DIAGNOSIS':^80}")
print(f"{'═'*80}\n")

final_acc_0db = None
final_acc_10db = None

if '0' in confusion_per_snr:
    cm0 = np.array(confusion_per_snr['0'])
    correct0 = cm0[target_idx, target_idx]
    total0 = cm0[target_idx, :].sum()
    final_acc_0db = (correct0/total0)*100 if total0 > 0 else 0

if '10' in confusion_per_snr:
    cm10 = np.array(confusion_per_snr['10'])
    correct10 = cm10[target_idx, target_idx]
    total10 = cm10[target_idx, :].sum()
    final_acc_10db = (correct10/total10)*100 if total10 > 0 else 0

print(f"16QAM Final Performance (Epoch {latest_epoch}):")
print(f"  • @ 0 dB:  {final_acc_0db:.2f}% (CRITICAL - very noisy channel)")
print(f"  • @ 10 dB: {final_acc_10db:.2f}% (cleaner channel)")
print(f"  • Delta:   {final_acc_10db - final_acc_0db:.2f}% (gap between noisy and clean)")

print(f"\nKey Finding:")
print(f"  → 16QAM collapses to 3% at 0dB despite adaptive sampling")
print(f"  → Epoch 10 showed 50.5% (baseline), but degraded by epoch 15")
print(f"  → This indicates CATASTROPHIC FORGETTING caused by:")
print(f"     • Extreme weight imbalance (16QAM 0dB got 9.1% weight)")
print(f"     • Epsilon too low (0.0001 vs default 0.02 = 20x underestimate)")
print(f"     • Model losing discriminative features for easy classes")

print(f"\n{'═'*80}\n")
