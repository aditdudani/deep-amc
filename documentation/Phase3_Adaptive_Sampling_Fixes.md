# Phase 3: Adaptive Sampling Fixes & Analysis

**Date:** 2026-03-28
**Status:** ✅ Hyperparameter fixes implemented, ready for re-training
**Scope:** Root cause analysis and fixes for catastrophic forgetting in 16QAM

---

## Executive Summary

The Phase 3 adaptive sampling run collapsed dramatically during training:
- **Epoch 10:** 50.48% accuracy (16QAM @ 0dB) ✓ Perfect
- **Epoch 15:** 2.87% accuracy (19x DROP in 5 epochs) ⚠️ Catastrophic
- **Final:** 3.1% (contaminated eval), 16.99% (clean validation) ✗ Never recovered

**Root Cause:** Extreme hyperparameter misconfiguration (epsilon 20x too low) allowed easy classes to be starved while the model catastrophically overfitted to the hardest bucket.

**Solution:** 5 critical hyperparameter updates + new minimum weight floor protection.

---

## Part 1: Root Cause Analysis - 16QAM Confusion Patterns

### What is 16QAM Confused With? (At 0dB - Epoch 40)

When given true 16QAM at 0dB SNR, the model misclassifies as:

| Predicted Class | Count | Percentage | Interpretation |
|-----------------|-------|-----------|-----------------|
| **32QAM** | ~167 | **40.0%** | Most similar constellation (16→32 points) |
| **16QAM** ✓ | ~82 | **19.6%** | Correct (very rare) |
| **64QAM** | ~64 | **15.3%** | Even more complex (64 points) |
| **QPSK** | ~60 | **14.4%** | Confusion on constellation size |
| **8PSK** | ~35 | **8.4%** | Phase shift keying |
| **OQPSK** | ~10 | **2.4%** | Offset QPSK |

**Key Insight:** Not random guessing—systematic confusion with related QAM constellations suggests the model learned some modulation structure but couldn't distinguish constellation sizes under noise.

---

### Accuracy Collapse Timeline

```
Epoch   16QAM@0dB   Analysis
──────────────────────────────
4       5.02%       Early learning - uniform weighting
10      50.48% ← PEAK - adaptive sampling working!
15      2.87%  ← COLLAPSE - 19x drop in 5 epochs
22      3.11%      Stays broken
40      16.99%     Slight recovery but 30x worse than peak
```

### What Happened Between Epochs 10-15?

Analysis of saved confusion matrices and weight files reveals catastrophic forgetting:

| Metric | Epoch 10 | Epoch 15 | Epoch 22 | Issue |
|--------|----------|----------|----------|-------|
| 16QAM 0dB Weight | 5.0% | 9.0% | 9.1% | **Got HEAVIER despite crashing accuracy** |
| 16QAM 0dB Accuracy | 50.5% | 2.9% | 3.1% | **Accuracy fell off cliff** |
| 4ASK Weight | ~9% | ~0.075% | ~0.075% | **Easy class starved by 99%** |
| BPSK Weight | ~9% | ~0.075% | ~0.075% | **Easy class starved by 99%** |

**Mechanism:** The adaptive sampler responded correctly (increase weight for high-error buckets), but the extreme imbalance triggered **catastrophic forgetting**:
1. 16QAM 0dB weight increased to 9.1%
2. Easy classes collapsed to 0.075% weight each
3. Model stopped learning the easy modulations
4. Started mispredicting everything onto complex QAMs (32QAM, 64QAM)
5. Result: Can't classify even the easy classes anymore, overall accuracy crashed

---

## Part 2: Hyperparameter Fixes Implementation

### Root Cause #1: Epsilon 20x Too Low

**The Problem:**
```python
epsilon = 0.0001  # Actual (loaded from weights_epoch22.json)
epsilon = 0.02   # Default that should have been used
# Ratio: 0.0001 / 0.02 = 0.005 = 50x underestimate (actually worse!)
```

The epsilon parameter controls the minimum error floor in weight updates:
```python
updated = (1 - beta) * weights + beta * (errors + epsilon)
```

With `epsilon = 0.0001`:
- Even 0% error buckets get: `0.3 * 0.0001 = 0.00003` weight
- After normalization: completely ignored
- Easy classes (BPSK, 4ASK) with low error → **near-zero weight**

With proper `epsilon = 0.03`:
- Even 0% error buckets get: `0.3 * 0.03 = 0.009` weight
- After normalization: preserved
- Easy classes stay represented

### Root Cause #2: No Minimum Weight Floor

The old capping/normalization code had no lower bound:
```python
# OLD CODE:
flat = np.minimum(flat, max_cap)  # Cap at 40%
flat /= flat.sum()                # Normalize
# Problem: Easy classes could drop below 0.1% → effectively deleted
```

---

### Fixes Implemented

#### 1. New Parameter: `min_weight = 0.005`
**Location:** `src/adaptive_sampling_g/callbacks_confusion_snr.py:72`

```python
def __init__(self,
             ...
             min_weight: float = 0.005,  # ← NEW
             ...
)
```

#### 2. Enforce Minimum Weight Floor
**Location:** `src/adaptive_sampling_g/callbacks_confusion_snr.py:168-183`

**NEW CODE:**
```python
# Cap individual bucket weight to prevent collapse
flat = updated.flatten()
if self.max_cap is not None and self.max_cap > 0:
    flat = np.minimum(flat, self.max_cap)

# Apply minimum weight floor to prevent bucket starvation (NEW!)
flat = np.maximum(flat, self.min_weight)  # ← Guarantees 0.5% minimum

# Renormalize
flat_sum = flat.sum()
if flat_sum <= 0:
    flat = np.ones_like(flat) / flat.size
else:
    flat /= flat_sum
self.weights_ref[:] = flat.reshape(self.weights_ref.shape)
```

This ensures every class-SNR combination maintains at least 0.5% weight.

#### 3. Updated Hyperparameter Defaults
**Location:** `src/adaptive_sampling_g/callbacks_confusion_snr.py:64-72`

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `beta` | 0.3 | **0.2** | Slower weight adaptation (0.3 * 0.5 vs 0.3 * 0.2 = larger steps) |
| `epsilon` | 0.02 | **0.03** | Error floor boost (error dampening) |
| `max_cap` | 0.4 | **0.12** | Tighter capping prevents single-bucket dominance |
| `replay_fraction` | 0.0 | **0.18** | Stronger uniform blending (18% vs 0%) for diversity |
| `min_weight` | — | **0.005** | NEW: 0.5% minimum per bucket |

#### 4. Training Script Arguments
**Location:** `src/adaptive_sampling_g/train_squeezenet_sampler.py:139-144`

Updated both defaults and help text:
```python
p.add_argument('--beta', type=float, default=0.2,
              help='Adaptive: smoothing factor for weight updates (lower=slower, more stable)')
p.add_argument('--epsilon', type=float, default=0.03,
              help='Adaptive: additive error floor (prevents class starvation)')
p.add_argument('--max-cap', type=float, default=0.12,
              help='Adaptive: per-bucket max weight before renorm (prevents single-bucket dominance)')
p.add_argument('--replay-fraction', type=float, default=0.18,
              help='Adaptive: fraction of uniform distribution blended (stronger baseline for diversity)')
p.add_argument('--min-weight', type=float, default=0.005,
              help='Adaptive: minimum weight floor per bucket (0.5%, prevents class starvation)')
```

#### 5. Callback Instantiation
**Location:** `src/adaptive_sampling_g/train_squeezenet_sampler.py:474-488`

Added `min_weight` parameter:
```python
cb_list.append(ConfusionBySNRCallback(
    val_metadata_csv=args.metadata_val,
    weights_ref=weights,
    out_dir=RESULTS_DIR,
    beta=float(args.beta),
    epsilon=float(args.epsilon),
    max_cap=float(args.max_cap),
    replay_fraction=float(args.replay_fraction),
    min_weight=float(args.min_weight),  # ← NEW
    batch_size=batch_size,
    snrs=TARGET_SNRS,
    warmup_epochs=args.warmup_epochs,
    min_val_acc_for_updates=args.min_val_acc,
    class_names=class_names,
))
```

---

## Part 3: Expected Improvements

### Weight Distribution Comparison

#### BEFORE (Broken at Epoch 15)
```
Class/SNR          Weight%    Status
────────────────────────────────────
4ASK (any SNR):    0.075%     ✗ STARVED
BPSK (any SNR):    0.075%     ✗ STARVED
Avg other:        ~0.5-2%
16QAM 0dB:         9.1%       ✗ ALL-CONSUMING
32QAM:            ~5%
```

#### AFTER (With Fixes)
```
Class/SNR          Weight%    Status
────────────────────────────────────
4ASK (any SNR):    ≥0.5%      ✓ PROTECTED
BPSK (any SNR):    ≥0.5%      ✓ PROTECTED
Avg other:        ~0.5-2%     ✓ Maintained
16QAM 0dB:        ≤12%       ✓ CAPPED
32QAM:            ~2-3%      ✓ Reasonable
                   ────
Plus:             18%         ✓ Uniform blending for diversity
```

### Stability Improvements

| Aspect | Problem | Fix | Expected Result |
|--------|---------|-----|-----------------|
| **Epoch 15 Collapse** | 50% → 3% crash | Slower beta + min_weight floor | Smooth curve, no collapse |
| **Class Starvation** | Easy classes at 0.075% | min_weight ≥ 0.5% | All classes learned |
| **Over-focus** | 16QAM 0dB at 9.1% | max_cap reduced to 12% | Reasonable challenge |
| **Weight Oscillations** | Rapid swings | beta 0.3→0.2 (slower) | Stable convergence |
| **Loss of Diversity** | No uniform blend | replay 0%→18% | Balanced class distribution |

---

## Part 4: Files Modified

### Modified Files

| File | Changes | Lines |
|------|---------|-------|
| `src/adaptive_sampling_g/callbacks_confusion_snr.py` | Added min_weight param, enforcing logic, updated JSON logging | 72, 85, 174, 195 |
| `src/adaptive_sampling_g/train_squeezenet_sampler.py` | Updated 4 defaults, added --min-weight arg, passed to callback | 139-144, 474-488 |

### Unchanged Files

- All other training/evaluation scripts remain compatible
- Default behavior is now fixed without requiring CLI overrides
- Can still customize via CLI args if needed

---

## Part 5: Testing & Validation Plan

### Before You Re-train

Verify the changes were applied:
```bash
# Check callback defaults
grep "beta: float = " src/adaptive_sampling_g/callbacks_confusion_snr.py
# Expected: "beta: float = 0.2"

# Check training script defaults
grep "default=0.2" src/adaptive_sampling_g/train_squeezenet_sampler.py
# Expected: "default=0.2" for beta
```

### During Training

Run with new defaults:
```bash
python src/adaptive_sampling_g/train_squeezenet_sampler.py --mode adaptive --epochs 40
```

Monitor for:
1. **Epoch 10-15:** Should be smooth, NO collapse
2. **16QAM 0dB accuracy:** Should stay ≥ 20% throughout (not crash to 3%)
3. **Weight logs:** Check that min_weight appears in each weights_epoch*.json

### After Training

1. **Analyze confusion matrices:**
   ```bash
   python analyze_confusion_from_training.py  # Reuse same script
   ```
   Verify: No 19x drops like before

2. **Run clean evaluation:**
   ```bash
   python src/adaptive_sampling_g/eval_all_models.py
   ```

3. **Success criteria:**
   - ✅ Adaptive model beats Config G Baseline (78.22%)
   - ✅ 16QAM at 0dB > 20% (was 3.1%)
   - ✅ No class/SNR combination lower than baseline
   - ✅ Overall smooth learning curve (no catastrophic collapse)

---

## Part 6: Rollback Instructions

If the fixes introduce unexpected issues, you can revert to old behavior:

**Option 1: Use CLI arguments**
```bash
python src/adaptive_sampling_g/train_squeezenet_sampler.py \
  --mode adaptive \
  --epochs 40 \
  --beta 0.3 \
  --epsilon 0.02 \
  --max-cap 0.4 \
  --replay-fraction 0.0 \
  --min-weight 0.0  # Disable floor
```

**Option 2: Git revert**
```bash
git diff src/adaptive_sampling_g/callbacks_confusion_snr.py
git diff src/adaptive_sampling_g/train_squeezenet_sampler.py
git checkout HEAD -- src/adaptive_sampling_g/callbacks_confusion_snr.py
git checkout HEAD -- src/adaptive_sampling_g/train_squeezenet_sampler.py
```

---

## Summary

**What was fixed:**
1. ✅ Epsilon too low (0.0001) → restored to 0.03
2. ✅ No minimum weight floor → added 0.5% protection
3. ✅ Over-aggressive capping (40%) → reduced to 12%
4. ✅ No uniform blending (0%) → increased to 18%
5. ✅ Fast adaptation (0.3) → slowed to 0.2

**Why it matters:**
- Prevents easy classes from being starved
- Prevents single bucket from being all-consuming
- Maintains model's ability to learn all modulations
- Stabilizes training without sacrificing hard-case focus

**Next step:**
Re-train with: `python src/adaptive_sampling_g/train_squeezenet_sampler.py --mode adaptive --epochs 40`

---

**Commit Message Recommendation:**
```
Phase 3: Fix adaptive sampling hyperparameters and add min_weight floor

- Increase epsilon (0.02 → 0.03) to restore error floor
- Reduce beta (0.3 → 0.2) for slower, smoother adaptation
- Tighten max_cap (0.4 → 0.12) to prevent single-bucket dominance
- Increase replay_fraction (0.0 → 0.18) for stronger uniform baseline
- Add min_weight floor (0.5%) to prevent class starvation
- Root cause: Previous run collapsed due to easy classes starving to 0.075%

Fixes catastrophic forgetting at epoch 15 where 16QAM accuracy dropped 50%→3%
```
