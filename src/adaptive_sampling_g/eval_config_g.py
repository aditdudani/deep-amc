#!/usr/bin/env python3
"""
Evaluate Config G Baseline (Phase 2 runner-up - K20 3x3 Cross Centered, single-channel).
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
        model_path='results_local/phase2_matrix/model_config_G.keras',
        metadata_val_csv='data/processed_g/metadata_val.csv',
        output_dir='results_local/phase3_clean_eval',
        model_name='Config_G_Baseline',
        batch_size=64
    )
