import h5py
import numpy as np
import os

def load_data_sample(file_path):
    expanded_path = os.path.expanduser(file_path)

    with h5py.File(expanded_path, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
        Z = hf['Z'][:]
    return X, Y, Z

if __name__ == '__main__':
    sample_path = '~/amc_project/data/RML2018.01A_sample.h5_last10k'

    try:
        X_sample, Y_sample, Z_sample = load_data_sample(sample_path)
        print("Data loaded successfully!")
        print(f"X shape: {X_sample.shape}")
        print(f"Y shape: {Y_sample.shape}")
        print(f"Z shape: {Z_sample.shape}")
        print(f"Number of records: {X_sample.shape}")
    except FileNotFoundError:
        print(f"ERROR: Sample file not found at {os.path.expanduser(sample_path)}")
        print("Please ensure you have downloaded the sample file from the server.")