# Backup Folder Verification Report

**Date:** 2026-04-02  
**Purpose:** Verify all important files from backup folders exist in `results/` before deletion

---

## Summary

| Backup Folder | Status | Safe to Delete? |
|---------------|--------|-----------------|
| `results_new/` | ✅ All files present | **YES** |
| `results_local_old/` | ✅ All files present | **YES** |
| `results_old/` | ⚠️ Phase 1 (Nov 2025) not copied | **YES** (if Phase 1 not needed) |

---

## Detailed Verification

### 1. `results_new/` (Intermediate Organized Structure)

**Result:** ✅ All files present in `results/`

Only "missing" file:
- `REORGANIZATION_SUMMARY.md` — Intentionally not added (you requested this)

**Action:** Safe to delete with `rm -rf results_new/`

---

### 2. `results_local_old/` (Original results_local/)

**Result:** ✅ All Phase 3 files present in `results/`

Files checked:
- `phase3_clean_eval/` evaluation results → copied to `results/adaptive_sampling_g/*/evaluation/`
- `phase2_matrix/` rankings → copied to `results/phase2_matrix/`
- Baseline evaluations → copied to `results/baselines/*/evaluation/`

**Action:** Safe to delete with `rm -rf results_local_old/`

---

### 3. `results_old/` (Original results/)

**Result:** ⚠️ 208 files not copied (intentionally)

#### Breakdown of "Missing" Files:

| Category | Count | Description |
|----------|-------|-------------|
| Phase 1 (Nov 2025) | 182 | `adaptive_sampling/`, `squeezenet/`, `evals/` runs from November 2025 |
| TensorFlow Events | 18 | Binary `events.out.tfevents.*` log files |
| Old SqueezeNet Results | 8 | `accuracy_by_snr_squeezenet.*` from Nov 2025 |

#### Phase 1 Files NOT Copied:
```
results_old/adaptive_sampling/20251115_201532/
results_old/adaptive_sampling/20251115_213633/
results_old/adaptive_sampling/20251115_223009/
results_old/adaptive_sampling/20251118_201834/
results_old/squeezenet/20251115_052516/
results_old/evals/squeezenet/20251119_214826/
results_old/evals/squeezenet/20251120_010240/
```

These are **Phase 1 experiments from November 2025** — before the Phase 3 adaptive sampling work.

**Action:** Safe to delete with `rm -rf results_old/` if you don't need Phase 1 experiments

---

## Phase 3 Completeness Check

All Phase 3 (Feb/Mar 2026) training runs verified:

| Timestamp | Model | Weights | Confusion | Logs | Eval |
|-----------|-------|---------|-----------|------|------|
| 20260224_084511 | ✅ | 40 | 37 | 2 | 0 |
| 20260224_124340 | ✅ | 8 | 5 | 1 | 0 |
| 20260224_130626 | ❌ (failed run) | 0 | 0 | 0 | 0 |
| 20260224_130702 | ✅ | 40 | 0 | 2 | 0 |
| 20260224_155039 | ✅ | 40 | 37 | 2 | 0 |
| 20260225_080634 | ✅ | 0 | 0 | 2 | 0 |
| 20260328_180649 | ✅ | 40 | 37 | 1 | 1 |

**Note:** `20260224_130626` was a failed/incomplete run (no model ever saved) — this is expected.

---

## Baselines & Comparisons

### Baselines
- `results/baselines/config_a/` — ✅ model.keras + 3 eval files
- `results/baselines/config_g/` — ✅ model.keras + 3 eval files

### Comparisons
- `results/comparisons/phase3_all_models/` — ✅ 7 files
  - `phase3_comparison_all_models.json`
  - `comparison_overall_accuracy.png`
  - `comparison_per_snr.png`
  - `comparison_per_class.png`
  - `comparison_heatmaps_all_models.png`
  - `comparison_delta_vs_baseline.png`
  - `16qam_0db_accuracy_trend.png`

---

## Deletion Commands

When ready, run:

```bash
rm -rf results_new/
rm -rf results_local_old/
rm -rf results_old/
```

Or all at once:
```bash
rm -rf results_new/ results_local_old/ results_old/
```

---

## What's Tracked in Git

After cleanup, `results/` contains:

**Tracked (JSON/CSV/PNG):**
- `results/adaptive_sampling_g/*/weights/*.json`
- `results/adaptive_sampling_g/*/confusion/*.json`
- `results/adaptive_sampling_g/*/logs/*.csv`
- `results/adaptive_sampling_g/*/evaluation/*.json` and `*.png`
- `results/baselines/*/evaluation/*`
- `results/comparisons/**/*`
- `results/phase2_matrix/*.json` and `*.csv`

**Ignored (.h5/.keras binaries):**
- `results/adaptive_sampling_g/*/model.h5`
- `results/baselines/*/model.keras`
- `results/phase2_matrix/*.keras`
