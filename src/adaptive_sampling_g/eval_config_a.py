#!/usr/bin/env python3
"""
Evaluate Config A Baseline (Phase 2 winner - K02 3x3 Cross, single-channel).
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
        model_path='results/baselines/config_a/model.keras',
        metadata_val_csv='data/processed_g/metadata_val.csv',
        output_dir='results/comparisons/phase3_all_models',
        model_name='Config_A_Baseline',
        batch_size=64
    )
