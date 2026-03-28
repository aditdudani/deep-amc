# 16QAM Confusion Analysis - Phase 3 Adaptive Sampling

**Generated:** 2026-03-28
**Training Run:** `20260224_155039` (Final adaptive sampling model)
**Analysis:** Confusion patterns from 40-epoch adaptive training

---

## Executive Summary

**The Problem:** 16QAM at 0dB SNR achieves only **3.1% accuracy** in final validation, despite being the target hardest-to-classify case.

**Root Cause:** **Catastrophic forgetting** due to extreme weight imbalance and severe hyperparameter misconfiguration.

**Key Evidence:**
- Epoch 10: 50.48% accuracy (healthy learning)
- Epoch 15: 2.87% accuracy (19x drop!)
- Epoch 40: 16.99% (eventual recovery but stayed low)

---

## What is 16QAM Confused With? (At 0dB - Epoch 40)

When given a true 16QAM signal at 0dB SNR, the model misclassifies it as:

| Predicted Class | Count | Percentage | Meaning |
|-----------------|-------|-----------|---------|
| **32QAM** | ~167 | **40.0%** | Similar 32-point constellation, slightly more complex |
| **16QAM** ✓ | ~82 | **19.6%** | Correct classification (very rare) |
| **64QAM** | ~64 | **15.3%** | 64-point constellation (finer grid, harder to distinguish) |
| **QPSK** | ~60 | **14.4%** | 4-point constellation (model confusing level count) |
| **8PSK** | ~35 | **8.4%** | Phase shift keying |
| **OQPSK** | ~10 | **2.4%** | Offset QPSK |

**Interpretation:**
- 40% of errors → 32QAM (most similar to 16QAM)
- Model struggles with distinguishing QAM constellation sizes under noise
- It's not completely random; there's systematic confusion with related modulations

---

## Accuracy Collapse: The Epoch Timeline

```
Epoch   16QAM@0dB   Status
────────────────────────────
4       5.02%       Early learning - all buckets getting uniform weight
10      50.48% ✓    PEAK - adaptive sampling is working!
15      2.87%  ⚠️   DROP by 19x in 5 epochs - catastrophic forgetting begins
22      3.11%       Continues to stay collapsed
40      16.99%      Slight recovery but still 30x worse than epoch 10
```

**What Happened Between Epochs 10-15?**

From the plan's analysis of weight dynamics:

| Metric | Epoch 10 | Epoch 15 | Epoch 22 | Issue |
|--------|----------|----------|----------|-------|
| 16QAM 0dB Weight | 5.0% | 9.0% | 9.1% | **Got HEAVIER, not lighter** |
| Accuracy | 50.5% | 2.9% | 3.1% | **Got WORSE despite heavy sampling** |
| Easy class weights (4ASK/BPSK) | ~9% | ~0.075% | ~0.075% | **Collapsed to near-zero** |

**Diagnosis:** The adaptive sampler responded to 16QAM's high error by:
1. ✓ Increasing 16QAM 0dB weight (correct action)
2. ✗ **Massacring easy classes** (4ASK/BPSK) down to 0.075% each
3. ✗ This extreme imbalance caused **catastrophic forgetting** - model lost discriminative features for  easy classes and started mispredicting everything onto complex QAMs

---

## Why Did This Happen?

### Root Cause #1: Epsilon Hyperparameter (20x Too Low)

From `weights_epoch22.json`:
```json
{
  "epsilon": 0.0001,  // ← ACTUAL (20x lower than default!)
  "beta": 0.3,
  "max_cap": 0.4,
  "replay_fraction": 0.1
}
```

The epsilon parameter controls the **minimum error floor** to prevent bucket starvation:
```python
updated = (1 - beta) * weights + beta * (errors + epsilon)
```

With `epsilon = 0.0001`:
- Even buckets with 0% error get weight `0.3 * 0.0001 = 0.00003`
- This is essentially ignored during normalization
- Easy classes (4ASK, BPSK) with low error → **near-zero weight**

With default `epsilon = 0.02`:
- Even buckets with 0% error get weight `0.3 * 0.02 = 0.006` (100x more!)
- Easy classes stay represented even with low error

### Root Cause #2: No Minimum Weight Floor

The capping/normalization has no **minimum weight guarantee**:
```python
# Current code:
flat = np.minimum(flat, max_cap)      # Cap at 40% per bucket
flat /= flat.sum()                    # Normalize

# Problem: Easy classes can drop below 0.5% → effectively ignored
```

**Solution:** Add `min_weight = 0.005` (0.5% per bucket) enforced AFTER capping, BEFORE normalization.

---

## Impact on Other Classes

### SNR 0dB Performance (Epoch 40)

| Class | Accuracy | Status |
|-------|----------|--------|
| 4ASK | 100% | ✓ Perfect (easy, binary) |
| BPSK | 100% | ✓ Perfect (easy, binary) |
| OQPSK | 14.7% | ⚠️ Dropped from baseline 14.7% → still problematic |
| **16QAM** | **3.1%** | ✗ **WORST** |
| 32QAM | 39.3% | ✗ Bad |
| 64QAM | 42.6% | ✗ Bad |
| 8PSK | 42.4% | ✗ Bad |
| QPSK | 22.3% | ✗ Bad |

**Key observation:** Complex modulationsare all suffering, while binary modulationsare untouched. Thissuggests:
- Model is focusing too hard on 16QAM 0dB (the hardest case)
- Overfitting to that specific bucket at expense of others
- The uniform replay wasn't strong enough (only 10%) to maintain diversity

---

## Recommendations (From Plan)

### Immediate Fixes (Task 3)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `epsilon` | 0.0001 | **0.02-0.05** | Restore error floor; prevents starvation |
| `beta` | 0.3 | **0.15-0.2** | Slower weight adaptation; avoid oscillations |
| `max_cap` | 0.4 | **0.10-0.15** | Prevent single bucket domination |
| `replay_fraction` | 0.1 | **0.15-0.20** | Stronger uniform baseline (20% vs 10%) |
| **NEW:** `min_weight` | — | **0.005** | Guarantee 0.5% minimum per bucket |

### Verification Strategy

After retraining with fixed hyperparameters:

1. **Check that epoch 10+ doesn't crash** - the catastrophic forgetting should disappear
2. **Monitor weight distribution** - should stay more uniform (easy classes ≥ 0.5%)
3. **Verify 16QAM 0dB doesn't collapse again** - target ≥ 50% at good epochs
4. **Validate vs baseline** - adaptive should beat Config G 78.22%, not regress

---

## Saved Artifacts

- `16qam_0db_accuracy_trend.png` - Trend line showing the collapse and recovery
- This analysis document

**Next Step:** Implement hyperparameter fixes in `callbacks_confusion_snr.py` and re-run training.
