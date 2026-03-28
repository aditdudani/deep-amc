# Phase 3 Clean Evaluation Scripts

These scripts evaluate models on the **validation split ONLY** using pre-generated PNG images and metadata CSVs. No HDF5 contamination—this is the correct and unbiased evaluation methodology.

## Files

### Main Script: `eval_validation_clean.py`
Generic evaluation script that can evaluate any model.

**Usage:**
```bash
python src/adaptive_sampling_g/eval_validation_clean.py \
  --model-path <path_to_model> \
  --model-name <descriptive_name> \
  [--metadata-val data/processed_g/metadata_val.csv] \
  [--output-dir results_local/phase3_clean_eval] \
  [--batch-size 64]
```

**Example:**
```bash
python src/adaptive_sampling_g/eval_validation_clean.py \
  --model-path results_local/phase2_matrix/model_config_A.keras \
  --model-name "Config_A_Baseline"
```

**Outputs:**
- `eval_<model_name>.json` - Full results (accuracy by SNR, per-class metrics, etc.)
- `eval_<model_name>.png` - Line plot of accuracy vs SNR
- `eval_<model_name>_heatmap.png` - Per-class per-SNR heatmap (8×6 matrix)

---

### Convenience Scripts

#### `eval_config_a.py`
Evaluate Config A Baseline (Phase 2 winner).
```bash
python src/adaptive_sampling_g/eval_config_a.py
# Outputs to: results_local/phase3_clean_eval/eval_Config_A_Baseline.*
```

#### `eval_config_g.py`
Evaluate Config G Baseline (Phase 2 runner-up).
```bash
python src/adaptive_sampling_g/eval_config_g.py
# Outputs to: results_local/phase3_clean_eval/eval_Config_G_Baseline.*
```

#### `eval_adaptive.py`
Evaluate Config G with Adaptive Sampling (weighted per-class per-SNR sampler).
```bash
python src/adaptive_sampling_g/eval_adaptive.py
# Outputs to: results_local/phase3_clean_eval/eval_Config_G_Adaptive.*
```

#### `eval_all_models.py`
Run all three evaluations and generate a comparison report.
```bash
python src/adaptive_sampling_g/eval_all_models.py
# Outputs individual results + comparison JSON to results_local/phase3_clean_eval/
```

---

## Evaluation Metrics

Each evaluation produces:

### Per-SNR Accuracy
- Accuracy for each SNR (0, 2, 4, 6, 8, 10 dB)
- Sample count per SNR
- Line plot showing accuracy trend

### Per-Class Overall Accuracy
- Accuracy for each of 8 classes (16QAM, 32QAM, 4ASK, 64QAM, 8PSK, BPSK, OQPSK, QPSK)
- Helps identify which classes are harder/easier

### Per-Class Per-SNR Heatmap
- 8×6 matrix showing accuracy at each class-SNR combination
- Useful for identifying problem areas (e.g., 16QAM at 0dB)

### Overall Accuracy
- Aggregate accuracy across entire validation set

---

## Data Requirements

The evaluation scripts require:
- **Model file**: Keras model (\.keras or \.h5)
- **Validation metadata CSV**: `data/processed_g/metadata_val.csv`
  - Columns: `file_path,class_name,class_id,snr,h5_index`
  - Points to pre-generated validation PNGs
- **Validation images**: Pre-generated PNGs in `data/processed_g/validation/`
  - Directory structure: `validation/<class>/<image.png>`

On your server, these files exist at:
```
data/processed_g/
├── metadata_train.csv
├── metadata_val.csv
├── metadata_summary.json
├── train/
│   ├── 16QAM/
│   ├── 32QAM/
│   └── ...
└── validation/
    ├── 16QAM/
    ├── 32QAM/
    └── ...
```

---

## Why This Evaluation is Correct

The old `phase3eval.py` script **evaluated on the full HDF5 file** (~90% training data), which contaminates results. This broke down as:
- Split into 90% train, 10% validation at preprocessing time
- Training images: generated from 90% of HDF5
- phase3eval.py: evaluated on full HDF5 (contamination!)

**These clean scripts fix this by:**
1. Using only pre-generated validation PNGs
2. Loading via metadata CSV that tracks exact indices
3. No HDF5 access → no contamination

---

## Comparison Usage

To get a side-by-side comparison of all three models:
```bash
python src/adaptive_sampling_g/eval_all_models.py
```

This runs `eval_config_a.py`, `eval_config_g.py`, and `eval_adaptive.py` in sequence and generates:
- Individual result files for each model
- Consolidated comparison JSON: `phase3_comparison_all_models.json`
- Console output with tables comparing:
  - Overall accuracy
  - Per-SNR accuracy
  - Per-class accuracy

---

## Next Steps (From Phase 3 Plan)

After running these clean evaluations:

1. **Analyze Results**: Identify if adaptive sampling truly improves or if it was just evaluation bias
2. **Check 16QAM at 0dB**: See if degradation persists with clean eval
3. **Run Task 4**: Fix hyperparameters (epsilon, beta, max_cap, replay_fraction)
4. **Re-train**: Train adaptive model with corrected hyperparameters
5. **Compare**: Re-evaluate with fixed hyperparameters to see if improvement holds

---

## File Organization

```
src/adaptive_sampling_g/
├── eval_validation_clean.py    # Main generic evaluation script
├── eval_config_a.py            # Conv.A baseline wrapper
├── eval_config_g.py            # Config G baseline wrapper
├── eval_adaptive.py            # Config G adaptive wrapper
├── eval_all_models.py          # Comprehensive comparison runner
└── EVAL_README.md              # This file
```

All three wrappers internally call `eval_validation_clean.py`.
