# Results Archive

This directory contains experimental and intermediate results from earlier phases and iterations.

## Contents

**Old Adaptive Sampling Results (pre-Phase 3)**
- `adaptive_sampling/` - Original adaptive sampling runs (KDE-based, not hardware-specific)

**Early Kernel Research (Phase 1)**
- `calibration/` - Shift parameter calibration experiments
- `kernel_debug/` - Kernel design debugging
- `kernel_search/` - Grid search iterations
- `hardware_test/` - Hardware feasibility tests
- `debug_hardware/` - Hardware image generation debugging

**Phase 2 Intermediate Results**
- `phase2_comparison/` - Intermediate channel comparison results (superseded by phase2_matrix)
- `all_graphs/` - Early visualization attempts

**Other Early Explorations**
- `googlenet/` - Early GoogLeNet baseline experiments
- `compare/` - Miscellaneous comparison runs
- `paper_figures/` - Draft figures
- `squeezenet/` - Early SqueezeNet baseline experiments

## Active Results

Active results are in the parent directory:
- `phase2_matrix/` - Phase 2 final kernel search results (7 configs, Pareto analysis)
- `phase3_baseline/` - Phase 3 baseline (Config A) evaluation on full HDF5
- `phase3_final/` - Phase 3 adaptive sampling training results (squeezenet_sampler_g_20260224_155039.h5)
- `phase3_clean_eval/` - **CURRENT** Clean validation-only evaluation (all 3 models vs validation split)
- `adaptive_sampling_g/` - **CURRENT** Config G adaptive sampling runs

## Notes

These archived results were key to development but should not be used for final reporting. Use only:
- `phase2_matrix/` for Phase 2 results
- `phase3_clean_eval/` for Phase 3 results (uncontaminated validation-only evaluation)
