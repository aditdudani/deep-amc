# Hyperparameter Fixes for Phase 3 Adaptive Sampling - Implementation Summary

**Date:** 2026-03-28
**Status:** ✅ IMPLEMENTED
**Impact:** Critical fixes to prevent catastrophic forgetting and weight starvation

---

## Problem Addressed

The previous adaptive sampling run (epoch 40) showed a dramatic collapse in 16QAM accuracy:
- Epoch 10: **50.48%** accuracy ✓
- Epoch 15: **2.87%** accuracy (19x drop) ⚠️
- Final: **3.1%** (contaminated), **16.99%** (clean validation)

**Root Cause:** Extreme hyperparameter misconfiguration allowed easy classes to be starved while the model focused catastrophically on the hardest bucket.

---

## Changes Made

### 1. **callbacks_confusion_snr.py** - Core Algorithm Updates

#### New Parameter Added
```python
min_weight: float = 0.005  # Minimum weight floor (0.5% per bucket)
```

#### Weight Update Algorithm (Lines 168-183)
**OLD BEHAVIOR:**
```python
# Cap to max_cap, then renormalize
flat = np.minimum(flat, max_cap)
flat /= flat.sum()  # ← Easy classes could drop below 0.1%
```

**NEW BEHAVIOR:**
```python
# 1. Cap to max_cap
flat = np.minimum(flat, max_cap)

# 2. ENFORCE MINIMUM WEIGHT FLOOR (NEW!)
flat = np.maximum(flat, min_weight)  # ← Guarantees 0.5% per bucket

# 3. Renormalize
flat /= flat.sum()
```

This ensures every class-SNR bucket gets at least 0.5% weight, preventing starvation of easy classes.

#### Hyperparameter Defaults

| Parameter | Old Default | New Default | Rationale |
|-----------|------------|-------------|-----------|
| `beta` | 0.3 | **0.2** | Slower weight adaptation reduces oscillations |
| `epsilon` | 0.02 | **0.03** | Error floor was being overridden; slight increase for stability |
| `max_cap` | 0.4 | **0.12** | Prevents single bucket from dominating (40% → 12%) |
| `replay_fraction` | 0.0 | **0.18** | Stronger uniform blending maintains class diversity |
| `min_weight` | — | **0.005** | NEW: Guarantees 0.5% minimum per bucket |

#### JSON Logging Updated
All weight files now include `min_weight` parameter for reproducibility:
```json
{
  "epoch": 22,
  "beta": 0.2,
  "epsilon": 0.03,
  "max_cap": 0.12,
  "min_weight": 0.005,
  "replay_fraction": 0.18,
  "weights": [...]
}
```

---

### 2. **train_squeezenet_sampler.py** - Training Script Updates

#### Argument Parser Changes (Lines 139-143)
```python
# OLD
p.add_argument('--beta', type=float, default=0.3, help='...')
p.add_argument('--epsilon', type=float, default=0.02, help='...')
p.add_argument('--max-cap', type=float, default=0.4, help='...')
p.add_argument('--replay-fraction', type=float, default=0.0, help='...')

# NEW
p.add_argument('--beta', type=float, default=0.2, help='Adaptive: smoothing factor for weight updates (lower=slower, more stable)')
p.add_argument('--epsilon', type=float, default=0.03, help='Adaptive: additive error floor (prevents class starvation)')
p.add_argument('--max-cap', type=float, default=0.12, help='Adaptive: per-bucket max weight before renorm (prevents single-bucket dominance)')
p.add_argument('--replay-fraction', type=float, default=0.18, help='Adaptive: fraction of uniform distribution blended in each update (stronger baseline for diversity)')
p.add_argument('--min-weight', type=float, default=0.005, help='Adaptive: minimum weight floor per bucket (0.5%, prevents class starvation)')
```

#### Callback Instantiation (Line 474)
Added `min_weight` parameter to callback:
```python
cb_list.append(ConfusionBySNRCallback(
    # ... existing args ...
    min_weight=float(args.min_weight),  # ← NEW
    # ... rest of args ...
))
```

---

## Expected Behavior After Fix

### Weight Distribution (Relative to broken run)

**Before (Epoch 15 collapse):**
- Easy classes: 0.075% weight (starved) → model forgets them
- 16QAM 0dB: 9.1% weight (extreme focus) → catastrophic memorization

**After (With new hyperparams):**
- Easy classes: ≥ 0.5% weight (protected minimum) → model remembers them
- 16QAM 0dB: Limited to ≤ 12% by max_cap (reasonable focus)
- Plus 18% blending with uniform (stronger diversity)

### Accuracy Stability

Expected improvements:
1. **No catastrophic collapse** - Epoch 15 crash should disappear
2. **Maintained class diversity** - Easy classes won't go to 0% weight
3. **Smoother learning curve** - Slower beta reduces oscillations
4. **Better generalization** - Min_weight floor prevents overfitting to single bucket

---

## Testing Plan

To validate these fixes:

1. **Run new training with defaults:**
   ```bash
   python src/adaptive_sampling_g/train_squeezenet_sampler.py \
     --mode adaptive \
     --epochs 40
   ```

2. **Check each weight file for:**
   - Min weight ≥ 0.005
   - Max weight ≤ 0.12
   - No bucket at exactly 0%

3. **Verify no epoch 15 collapse:**
   - Create trend plot from confusion matrices
   - 16QAM 0dB should NOT exceed 50% then drop to 3%

4. **Compare to baseline:**
   ```bash
   python src/adaptive_sampling_g/eval_all_models.py
   ```
   - Config G Adaptive should beat Config G Baseline (78.22%)
   - 16QAM should improve over baseline

---

## Rollback Instructions

If issues occur, revert to old defaults:

```bash
git diff src/adaptive_sampling_g/callbacks_confusion_snr.py
git diff src/adaptive_sampling_g/train_squeezenet_sampler.py
```

Or run with custom args to override:
```bash
python src/adaptive_sampling_g/train_squeezenet_sampler.py \
  --beta 0.3 \
  --epsilon 0.02 \
  --max-cap 0.4 \
  --replay-fraction 0.0 \
  --min-weight 0.0  # Disable floor to revert
```

---

## Files Modified

1. ✅ `src/adaptive_sampling_g/callbacks_confusion_snr.py`
   - Added `min_weight` parameter
   - Updated weight algorithm to enforce minimum floor
   - Updated JSON logging

2. ✅ `src/adaptive_sampling_g/train_squeezenet_sampler.py`
   - Updated 4 hyperparameter defaults
   - Added `--min-weight` argument
   - Passed to callback

---

## Phase 3 Progress

| Task | Status | Evidence |
|------|--------|----------|
| Analyze confusion | ✅ Complete | `16QAM_CONFUSION_ANALYSIS.md`, trend plot |
| Fix hyperparameters | ✅ Complete | This document |
| Re-run training | ⏳ Next | Will use new defaults |
| Re-evaluate | ⏳ Next | Compare to baselines |

---

## Next Steps

1. **Train new adaptive model** with corrected hyperparameters
2. **Analyze confusion matrices** from new training run
3. **Run clean evaluation** on new adaptive model
4. **Compare vs baselines** (Config A: 75.70%, Config G: 78.22%)
5. **Document final results**

---

*Implemented as part of Phase 3 hyperparameter tuning based on extensive confusion matrix analysis.*
