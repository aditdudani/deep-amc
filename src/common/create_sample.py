import h5py
import numpy as np
import os

SOURCE_FILE_PATH = '/app/data/GOLD_XYZ_OSC.0001_1024.hdf5'
SAMPLE_FILE_PATH = '/app/data/RML2018.01A_sample.h5'
NUM_SAMPLES = 10000

def create_data_sample():
    print(f"Reading from source file: {SOURCE_FILE_PATH}")
    try:
        with h5py.File(SOURCE_FILE_PATH, 'r') as source_file, h5py.File(SAMPLE_FILE_PATH, 'w') as sample_file:
            total_records = source_file['X'].shape[0]
            print(f"Slicing last {NUM_SAMPLES} records from /X (total {total_records})...")
            x_data = source_file['X'][-NUM_SAMPLES:]
            sample_file.create_dataset('X', data=x_data)
            y_data = source_file['Y'][-NUM_SAMPLES:]
            sample_file.create_dataset('Y', data=y_data)
            z_data = source_file['Z'][-NUM_SAMPLES:]
            sample_file.create_dataset('Z', data=z_data)
            print("Sample file created:", SAMPLE_FILE_PATH)
            print("Records:", len(x_data))
    except FileNotFoundError:
        print(f"ERROR: Source file not found at {SOURCE_FILE_PATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    create_data_sample()
