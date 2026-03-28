# Deleted Experimental Artifacts

This document records all experimental code deleted during the Phase 3 cleanup. Each entry includes what was learned, why it was rejected, and how to restore it if needed.

---

## Root Duplicate Files (Removed Feb 24, 2026)

These were exact copies or near-duplicates of files in `src/common/` and specialized subdirectories. Keeping them caused import confusion and maintenance burden.

### `src/squeezenet.py`
- **What it did:** Defined SqueezeNet v1.1 architecture (Fire modules, model builder)
- **Why it existed:** Early copy before `common/squeezenet.py` was established
- **Why we don't need it:** All active code imports from `src/common/squeezenet.py`
- **Why it was rejected:** Duplicate; only used by deleted `train_squeezenet.py`
- **What we learned:** Centralize reusable components in `common/` to avoid maintenance duplication
- **How to restore:**
  ```bash
  git show HEAD~N:src/squeezenet.py > src/squeezenet.py  # (replace N with commits back)
  # Or: git checkout <commit-hash> -- src/squeezenet.py
  ```

### `src/model_builder.py`
- **What it did:** Built InceptionV3 transfer learning models for baseline
- **Why it existed:** Early copy before common module structure
- **Why we don't need it:** Never imported; all code uses `src/common/model_builder.py`
- **Why it was rejected:** Unused duplicate with slightly different comments
- **What we learned:** Import from `common` consistently across all modules
- **How to restore:**
  ```bash
  git show HEAD~N:src/model_builder.py > src/model_builder.py
  ```

### `src/image_generator.py`
- **What it did:** TensorFlow I/Q to 3-channel image generation (KDE-based with alpha parameters)
- **Why it existed:** Early iteration before centralization
- **Why we don't need it:** All code uses `src/common/image_generator.py`
- **Why it was rejected:** Root version had different imports; common version is the canonical implementation
- **What we learned:** Keep image generation pipeline centralized to prevent drift between baselines
- **How to restore:**
  ```bash
  git show HEAD~N:src/image_generator.py > src/image_generator.py
  ```

### `src/data_loader.py`
- **What it did:** Lazy-loading HDF5 data (loads 176k+ samples without RAM explosion)
- **Why it existed:** Early copy before common structure
- **Why we don't need it:** Root version had eager imports; `src/common/data_loader.py` is cleaner
- **Why it was rejected:** Unused; common version is the standard
- **What we learned:** Shared I/O utilities should be in common
- **How to restore:**
  ```bash
  git show HEAD~N:src/data_loader.py > src/data_loader.py
  ```

### `src/create_sample.py`
- **What it did:** Created smaller sample HDF5 files from full RadioML dataset
- **Why it existed:** Utility script, early copy
- **Why we don't need it:** Root version unused; common version available if needed
- **Why it was rejected:** Duplicate utility
- **What we learned:** Keep dataset utilities in common/
- **How to restore:**
  ```bash
  git show HEAD~N:src/create_sample.py > src/create_sample.py
  ```

### `src/preprocess.py`
- **What it did:** Offline preprocessing pipeline - generates training/validation PNGs from HDF5 IQ samples
- **Why it existed:** Early copy before common module
- **Why we don't need it:** Root version uses loose imports; `src/common/preprocess.py` is the standard
- **Why it was rejected:** Unused; common version is the canonical pipeline
- **What we learned:** Image preprocessing should be centralized and imported consistently
- **How to restore:**
  ```bash
  git show HEAD~N:src/preprocess.py > src/preprocess.py
  ```

---

## Obsolete Baseline Training Scripts

These training scripts were superseded by organized baselines in subdirectories.

### `src/train.py` (InceptionV3 Duplicate)
- **What it did:** InceptionV3 training with warmup + SGD fine-tuning (45 epochs, ImageNet transfer learning)
- **Why it existed:** Early standalone baseline before `baseline_peng/` was established
- **Why we don't need it:** Functionally identical to `src/baseline_peng/train_inceptionv3.py` (only missing sys.path setup)
- **Why it was rejected:** Duplicate baseline; canonicalized in dedicated `baseline_peng/` directory
- **What we learned:** Organize baselines into dedicated directories with clear naming
- **How to restore:**
  ```bash
  git show HEAD~N:src/train.py > src/train.py
  ```

### `src/train_squeezenet.py` (SqueezeNet Duplicate)
- **What it did:** SqueezeNet v1.1 training with RMSprop + ReduceLROnPlateau (40 epochs)
- **Why it existed:** Early standalone before `baseline_chahil/` was established
- **Why we don't need it:** Functionally similar to `src/baseline_chahil/train_squeezenet.py`, imports broken (depended on deleted `squeezenet.py`)
- **Why it was rejected:** Depends on deleted root copies; canonical baseline in `baseline_chahil/`
- **What we learned:** Consolidate baselines into named directories for clarity and maintainability
- **How to restore:**
  ```bash
  git show HEAD~N:src/train_squeezenet.py > src/train_squeezenet.py
  # WARNING: Will fail to import squeezenet.py - requires restoring that file too
  ```

---

## Experimental Phase 2 Architecture Search

These were intermediate drafts during Phase 2 (kernel architecture matrix search).

### `src/phase2_channel_comparison.py`
- **What it did:** Single-channel vs multi-channel architecture comparison (proof-of-concept)
- **Why it existed:** Initial Phase 2 exploration on 10% dataset, 10 epochs
- **Why we don't need it:** Superseded by rigorous Phase 2 study
- **Why it was rejected:** Limited scope (10% data, 10 epochs); final Phase 2 used 100% data, 40 epochs across 7 configs
- **What we learned:** Full-scale evaluation on complete dataset is essential before finalizing conclusions
- **How to restore:**
  ```bash
  git show HEAD~N:src/phase2_channel_comparison.py > src/phase2_channel_comparison.py
  ```

### `src/phase2_channel_comparison_v2.py`
- **What it did:** Rigorous Phase 2 comparison (full dataset, 40 epochs, single/dual/triple channel configs)
- **Why it existed:** v2 attempted to fix scope issues from v1
- **Why we don't need it:** Superseded by final `src/phase2_architecture_matrix.py` (Feb 24, more comprehensive)
- **Why it was rejected:** v2 was intermediate work; final matrix study expanded to 7 configs with Pareto analysis
- **What we learned:** The final matrix approach (7 configs + Pareto) was the decisive methodology
- **How to restore:**
  ```bash
  git show HEAD~N:src/phase2_channel_comparison_v2.py > src/phase2_channel_comparison_v2.py
  ```
- **Note:** Results from this run went into `results_local/phase2_comparison/`; Phase 2 final results are in `results_local/phase2_matrix/`

---

## Experimental Phase 1 Hardware Exploration

These explored early hardware image generation approaches before settling on Config G.

### `src/calibrate_hardware.py`
- **What it did:** Phase 1 shift calibration sweep - tested different bit-shift values (shift_val) for hardware kernels
- **Context:** FPGA-compatible integer-only image generation with fixed GAIN and dynamic shifts
- **Why it existed:** Early attempt to find optimal hardware shift parameters for 3x3 cross kernel
- **Why we don't need it:** Phase 1 final work (`kernel_grid_search.py`) tested 23 kernel configurations systematically; shift calibration was subsumed
- **Why it was rejected:** Narrow focus on single parameter; final Phase 1 grid search method was more comprehensive
- **What we learned:** Systematic grid search across kernel size/type is more robust than manual shift calibration
- **How to restore:**
  ```bash
  git show HEAD~N:src/calibrate_hardware.py > src/calibrate_hardware.py
  ```

### `src/simulate_hardware.py`
- **What it did:** FPGA hardware digital twin/emulator with 3-channel image generation
  - Used 3x3 sharp kernel, 3x3 medium kernel, 11x11 uniform blur kernel
  - Integer-only math, fixed GAIN=128, bit-shifts for output
  - Tested hardware-generated images against SqueezeNet baseline
- **Context:** Early exploration of multi-channel hardware kernels for AMC
- **Why it existed:** Proof-of-concept FPGA simulation before hardware implementation
- **Why we don't need it:** Superseded by Config G (1-channel center-weighted cross kernel)
  - Config G simpler (1 channel, not 3)
  - Config G uses proven cross pattern (Phase 1 winner was 3x3 cross, single-channel)
  - Config G chosen for Phase 3 validation
- **Why it was rejected:** Multi-channel approach added FPGA complexity without accuracy benefit; single 3x3 cross performed better (Phase 2 Pareto analysis)
- **What we learned:**
  - Single-channel 3x3 cross achieves 97.9% of max accuracy at 2.9% memory cost (much better than 3-channel)
  - FPGA kernels must balance accuracy gain vs memory/DSP cost; simplicity often wins
- **How to restore:**
  ```bash
  git show HEAD~N:src/simulate_hardware.py > src/simulate_hardware.py
  ```

### `src/proper_hw_train_test.py`
- **What it did:** Training test on hardware-generated images - verified if hardware-generated images were learnable by SqueezeNet
- **Context:** Test harness for hardware image generation pipeline
- **Why it existed:** Debugging script to validate hardware image quality before full evaluation
- **Why we don't need it:** Debugging artifact; confirmed images were learnable; not needed going forward
- **Why it was rejected:** One-off test script; moved to continuous validation in training pipeline
- **What we learned:** Hardware image generation can produce learnable patterns for AMC without degradation
- **How to restore:**
  ```bash
  git show HEAD~N:src/proper_hw_train_test.py > src/proper_hw_train_test.py
  ```

### `src/quick_hw_train_test.py`
- **What it did:** Quick in-memory test of hardware images - loaded pre-generated images into RAM, tested multiple optimizers (Adam, SGD) with different learning rates on SqueezeNet
- **Context:** Rapid debugging/validation script
- **Why it existed:** Quick iteration on optimizer tuning for hardware image training
- **Why we don't need it:** Debugging artifact; optimizer selection is now part of formal training scripts
- **Why it was rejected:** One-off diagnostic; insights captured in training pipeline
- **What we learned:** SGD with momentum worked better than Adam for hardware images (consistent with baseline findings)
- **How to restore:**
  ```bash
  git show HEAD~N:src/quick_hw_train_test.py > src/quick_hw_train_test.py
  ```

### `src/test_gaussian_blur.py`
- **What it did:** Experimental exploration of Gaussian blur kernels for hardware image generation
- **Context:** Phase 1 kernel exploration - testing alternative blur patterns
- **Why it existed:** Early kernel design iteration, exploring Gaussian as alternative to cross/uniform patterns
- **Why we don't need it:** Gaussian blur was rejected during Phase 1 grid search
- **Why it was rejected:**
  - Gaussian blur insufficient for spatial resolution at low SNR (0-2 dB) where cross kernel excels
  - Cross kernel's sharp edges better preserve constellation boundaries
  - Phase 1 grid search proved 3x3 cross superior (highest accuracy/cost ratio)
- **What we learned:**
  - Gaussian blur too smooth for AMC constellation classification
  - Sharp, cross-shaped kernels better preserve modulation structure
  - Grid search methodology proved more effective than single exploratory scripts
- **How to restore:**
  ```bash
  git show HEAD~N:src/test_gaussian_blur.py > src/test_gaussian_blur.py
  ```

---

## Obsolete Evaluation Scripts (Removed Mar 28, 2026)

These evaluation scripts were superseded by clean validation-only evaluation infrastructure.

### `src/evaluate.py`
- **What it did:** InceptionV3 evaluation script with hardcoded paths
  - Evaluated on full HDF5 dataset (196,608 samples per epoch)
  - Generated 3-channel images on-the-fly using KDE-based alpha parameters (10.0, 1.0, 0.1)
  - Limited evaluation to 200 samples/class/SNR for runtime/memory reasons
  - Output per-SNR accuracy JSON + visualization PNG
- **Why it existed:** Early evaluation framework before validation-only evaluation was established
- **Why we don't need it:**
  - **CONTAMINATED EVALUATION:** Evaluated on full HDF5 (~90% training data), not on held-out validation split
  - Superseded by `src/adaptive_sampling_g/eval_validation_clean.py` which evaluates only on validation PNGs (10% held-out)
  - Used alpha-based image generation (not Config G hardware kernels)
- **Why it was rejected:**
  - Results were misleading - reported ~82% overall accuracy by evaluating on training data
  - Clean validation-only evaluation (using held-out 10%) reports ~75-79% (realistic)
  - Hard-coded paths made it inflexible for different models
  - Per-SNR evaluation was limited to 200 samples/class/SNR; full validation has 3,000+ per bucket
- **What we learned:**
  - Evaluation must use held-out validation split, NOT full dataset
  - Training logs with Keras validation metrics are more trustworthy than post-hoc evaluation scripts
  - Contaminated evaluation can hide true model weaknesses (e.g., 16QAM 0dB dropped from 50.5% to 3.1% - only visible with proper eval)
- **How to restore:**
  ```bash
  git show df2ee84:src/evaluate.py > src/evaluate.py
  ```
- **Replacement:** `src/adaptive_sampling_g/eval_validation_clean.py` (clean, uses validation metadata CSV)

### `src/eval_squeezenet_by_snr.py`
- **What it did:** SqueezeNet evaluation with flexible CLI arguments
  - Evaluated on full HDF5 (contaminated evaluation, same issue as evaluate.py)
  - Generated 3-channel images using KDE alphas (10.0, 1.0, 0.1)
  - Supported CLI options: custom HDF5 path, model path, per-class matrix output, streaming chunk sizes
  - Output per-SNR accuracy JSON + optional per-class per-SNR matrix
- **Why it existed:** Early generalized evaluation for SqueezeNet baseline
- **Why we don't need it:**
  - **CONTAMINATED EVALUATION:** Same issue as evaluate.py - evaluated on ~90% training data
  - Superseded by `src/adaptive_sampling_g/eval_validation_clean.py` which is model-agnostic and uses validation-only split
  - Used alpha-based architecture (not Config G hardware kernels)
- **Why it was rejected:**
  - Full HDF5 evaluation masked true performance (realistic ~75-79% vs inflated ~82-83%)
  - Per-class matrix was useful but can be extracted from clean eval JSON output
  - Flexible CLI arguments added complexity without corresponding accuracy benefit
- **What we learned:**
  - Flexibility in evaluation parameters (alphas, chunk sizes, batch sizes) is less critical than evaluation integrity (held-out split)
  - Simple, uncontaminated evaluation is better than feature-rich contaminated evaluation
- **How to restore:**
  ```bash
  git show df2ee84:src/eval_squeezenet_by_snr.py > src/eval_squeezenet_by_snr.py
  ```
- **Replacement:** `src/adaptive_sampling_g/eval_validation_clean.py` (clean, works on any Keras model)

---

## Summary of Learnings

| Category | Key Insight |
|----------|-------------|
| **Code Organization** | Centralize reusable components in `common/`; dedicate directories to baselines; remove root duplicates |
| **Experimental Methodology** | Full-scale evaluation (100% data, full epochs) more reliable than quick proofs-of-concept |
| **Phase 1** | Systematic grid search (23 kernels) beats manual parameter tuning; 3x3 cross single-channel wins |
| **Phase 2** | Pareto analysis crucial; single-channel achieves 97.9% accuracy at 2.9% memory cost vs multi-channel |
| **Hardware Design** | Simplicity beats complexity; fixed bit-widths better than dynamic scaling; cross kernel > Gaussian blur |
| **Evaluation** | Per-SNR and per-class metrics essential; training on hardware images validates feasibility |

---

## Restoration Guide

To restore any deleted file:

```bash
# Find the commit that deleted it (check git log)
git log --oneline --diff-filter=D -- src/<filename>

# Restore from specific commit (one commit before deletion)
git show <commit-hash>^:src/<filename> > src/<filename>

# Or restore entire directory state at a commit
git checkout <commit-hash> -- src/
```

Example:
```bash
# Find when quick_hw_train_test.py was deleted
git log --oneline src/quick_hw_train_test.py

# Output: abc1234 deleted quick_hw_train_test.py
# Restore it
git show abc1234^:src/quick_hw_train_test.py > src/quick_hw_train_test.py
```

---

**Cleanup Commit:** df2ee84bd06a727f967983251b920ecc5da4c13a
- **Date:** Mar 28, 2026, 5:29 PM IST
- **Files Deleted:** 15
- **Lines Removed:** 3,420
- **Author:** Adit Dudani (aditdudani@gmail.com)
- **Message:** "Code cleanup, deleted redundant files"

**Evaluation Scripts Deletion:** Mar 28, 2026 (Phase 3 post-eval)
- **Files Deleted:** 2 (evaluate.py, eval_squeezenet_by_snr.py)
- **Reason:** Contaminated evaluation (full HDF5 instead of validation-only split)
- **Replacement:** `src/adaptive_sampling_g/eval_validation_clean.py` + `eval_all_models.py`

**Last Updated:** Mar 28, 2026 (Phase 3 eval infrastructure completion)
