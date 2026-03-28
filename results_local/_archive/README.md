# Results Archive

This directory contains results from earlier research phases. Organized into **foundational work** (useful, contributed to understanding) and **exploratory work** (intermediate attempts, not final).

---

## Foundational Work (Contributed to Understanding & Reports)

These results were important for research progression and appear in published comparisons.

### Software Adaptive Sampling (Original Implementation)
- `adaptive_sampling/` - Original software-based adaptive sampling
  - `adaptive_v1/` - Initial implementation with KDE-based image generation
  - `adaptive_v2/` - Refined version with improved weight update logic
  - Both used alpha parameters (10.0, 1.0, 0.1) for 3-channel image generation
  - Results compared against software baseline in research report

### Baseline Comparisons & Architectures
- `compare/` - Comparison visualizations of software adaptive sampling vs software baseline
  - Shows accuracy vs SNR comparisons across different modulations
  - Used in methodology validation
- `googlenet/` - Early GoogLeNet baseline exploration (validation of architecture selection process)
- `squeezenet/` - Early SqueezeNet baseline exploration (alternative to InceptionV3)

### Phase 1: Kernel Search (Grid Search of 23 Kernels)
- `kernel_search/` - Phase 1 grid search data
  - Contains kernel rankings from multiple runs (K1-K19 kernels)
  - Multiple iterations tracked in timestamped CSV/JSON files
  - Final results in parent `phase2_matrix/` directory
- `all_graphs/` - Visualizations from Phase 1 and Phase 2
  - kernel_*.png files show Phase 1 comparison (K1-K19 kernels)
  - phase2_*.png files show Phase 2 comparison (Configs A-F, 7 final candidates)
  - Useful for kernel design understanding and Pareto analysis
- `kernel_debug/` - Kernel visualization utilities
  - Useful for debugging kernel designs and understanding filter behavior

---

## Exploratory Work (Intermediate Approaches, Not Final)

These were exploratory attempts that were either superseded or abandoned.

### Hardware Exploration (Pre-Config G)
- `calibration/` - Shift parameter calibration attempts
  - Tested different bit-shift values for hardware kernels
  - Superseded by systematic `kernel_grid_search.py` approach in Phase 1
- `hardware_test/` - Early hardware feasibility tests
  - Validated that hardware-generated images were learnable
  - Debugging artifact; insights moved into training pipeline
- `debug_hardware/` - Hardware image generation debugging
  - Multi-channel hardware simulation (3x3 sharp, 3x3 medium, 11x11 blur)
  - Explored multi-channel approaches eventually superseded by Config G (single-channel 3x3 cross)

### Phase 2 Intermediate Results
- `phase2_comparison/` - Intermediate Phase 2 runs (v1 and v2)
  - `20260203_202046/` - First rigorous comparison (full dataset, 40 epochs, multiple channels)
  - `20260203_214223/` - Second iteration with refinements
  - Both superseded by final `phase2_architecture_matrix.py` (parent directory)
  - Expanded to 7 final configs (A-F) with full Pareto analysis

### Work-In-Progress
- `paper_figures/` - Draft figures and early visualization attempts
  - Pre-publication versions, not final figures

---

## Active Results (In Parent Directory)

Use these for reporting and analysis:
- `phase2_matrix/` - **Phase 2 Final**: Kernel search results for 7 configs (A-F) with Pareto analysis
- `phase3_baseline/` - **Phase 3 Baseline**: Config-A baseline evaluation
- `phase3_final/` - **Phase 3 Training**: Adaptive sampling model outputs
- `phase3_clean_eval/` - **Phase 3 Final Eval**: ✅ Clean validation-only evaluation (all 3 models)
- `adaptive_sampling_g/` - **Current Work**: Config G adaptive sampling training logs

---

## Summary

- **Archive Size**: 20 MB of historical work
- **Useful for Understanding**: Yes - shows research progression and methodology validation
- **Use for Final Reports**: No - use only active results above
- **Restore Individual Results**: Git can restore any file; see git log for specific commits
