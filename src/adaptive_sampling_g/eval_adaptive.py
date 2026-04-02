#!/usr/bin/env python3
"""
Evaluate Config G with Adaptive Sampling (per-class per-SNR weighted sampler).
Uses validation split ONLY (no HDF5 contamination).
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from eval_validation_clean import eval_validation_clean

if __name__ == '__main__':
    eval_validation_clean(
        model_path='results/adaptive_sampling_g/20260328_180649/model.h5',  # Update to latest
        metadata_val_csv='data/processed_g/metadata_val.csv',
        output_dir='results/comparisons/phase3_all_models',
        model_name='Config_G_Adaptive',
        batch_size=64
    )
