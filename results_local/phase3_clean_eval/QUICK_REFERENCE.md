# Quick Reference - Hyperparameter Fixes

## What Changed?

### The Problem
Training collapsed at epoch 15:
```
Epoch 10:  50.5% accuracy (16QAM @ 0dB) ✓ Good!
Epoch 15:  2.9% accuracy (16QAM @ 0dB)  ⚠️ DROPPED 19x in 5 epochs
Epoch 22:  3.1% final accuracy           ✗ Stayed broken
```

**Why:** Easy classes starved to 0.075% weight while hard class got 9.1%

---

## The Fix

### Hyperparameter Changes

```
OLD DEFAULTS          NEW DEFAULTS
─────────────────────────────────────
beta:     0.3    →    0.2       (slower learning)
epsilon:  0.02   →    0.03      (error floor boost)
max_cap:  0.4    →    0.12      (cap individual buckets tighter)
replay:   0.0    →    0.18      (stronger uniform baseline)
min_weight: ✗    →    0.005     (NEW: protect easy classes)
```

### The Key Innovation: Minimum Weight Floor

**NEW CODE** in weight update (callbacks_confusion_snr.py line 174):
```python
flat = np.maximum(flat, self.min_weight)  # Guarantee 0.5% minimum per bucket
```

This ensures:
- Easy classes (BPSK, 4ASK) never drop below 0.5%
- Model can't forget basic features
- Hard class (16QAM 0dB) is challenged, not all-consuming

---

## Files Changed

| File | Changes | Impact |
|------|---------|--------|
| `callbacks_confusion_snr.py` | +min_weight param, +enforcing logic | Core algorithm |
| `train_squeezenet_sampler.py` | +--min-weight arg, updated 4 defaults | Training entrypoint |

---

## Before vs After

### Weight Distribution (approximate)

#### BEFORE (broken):
```
Class/SNR      Weight%
─────────────────────
4ASK:          0.075% ← STARVED
BPSK:          0.075% ← STARVED
16QAM 0dB:     9.1%   ← ALL-CONSUMING
32QAM:        ~5%
... etc
```

#### AFTER (fixed):
```
Class/SNR      Weight%
─────────────────────
4ASK:          ≥0.5%  ← PROTECTED
BPSK:          ≥0.5%  ← PROTECTED
16QAM 0dB:     ≤12%   ← CAPPED
32QAM:        ~2-3%
... + 18% blended uniform baseline
```

---

## How to Re-train

```bash
# With new defaults (recommended):
python src/adaptive_sampling_g/train_squeezenet_sampler.py --mode adaptive --epochs 40

# Or with custom settings:
python src/adaptive_sampling_g/train_squeezenet_sampler.py \
  --mode adaptive \
  --epochs 40 \
  --beta 0.2 \
  --epsilon 0.03 \
  --max-cap 0.12 \
  --replay-fraction 0.18 \
  --min-weight 0.005
```

---

## Success Criteria

✅ Epoch 10-15: No catastrophic collapse (should be smooth)
✅ Epoch 40: 16QAM @ 0dB > 20% (not 3%)
✅ Final eval: Accuracy > 78.22% (Config G baseline)
✅ No class has 0% weight (all have ≥0.5%)

---

## Documentation

📄 Full details: `HYPERPARAMETER_FIXES_SUMMARY.md`
📄 Confusion analysis: `16QAM_CONFUSION_ANALYSIS.md`
📊 Trend plot: `16qam_0db_accuracy_trend.png`

---

**Status:** ✅ READY TO RE-TRAIN

Next: Run training, analyze results, compare to baseline.
