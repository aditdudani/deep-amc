import h5py
import numpy as np
import os

# --- Configuration ---
# Define the absolute paths inside the Docker container
SOURCE_FILE_PATH = '/app/projects/amc_project/data/GOLD_XYZ_OSC.0001_1024.hdf5'
SAMPLE_FILE_PATH = '/app/projects/amc_project/data/RML2018.01A_sample.h5'

# Number of records to include in the sample
NUM_SAMPLES = 10000

# --- Script ---
def create_data_sample():
    """
    Creates a smaller sample HDF5 file from the large RadioML 2018.01A dataset.
    This function reads the first NUM_SAMPLES records from the X, Y, and Z
    datasets without loading the entire file into memory.
    """
    print(f"Reading from source file: {SOURCE_FILE_PATH}")

    try:
        # Open the source file in read mode ('r') and the new sample file in write mode ('w')
        with h5py.File(SOURCE_FILE_PATH, 'r') as source_file, h5py.File(SAMPLE_FILE_PATH, 'w') as sample_file:
            
            # --- Process X dataset (Signal Data) ---
            print(f"Slicing first {NUM_SAMPLES} records from /X...")
            # Get a handle to the full dataset on disk
            x_data = source_file['X'][:NUM_SAMPLES]
            # Read only the first NUM_SAMPLES records into memory
            x_sample = x_data
            # Create a new dataset in the sample file and write the sliced data
            sample_file.create_dataset('X', data=x_sample)

            # --- Process Y dataset (Labels) ---
            print(f"Slicing first {NUM_SAMPLES} records from /Y...")
            # ** THE FIX IS HERE: ** Get a handle to the 'Y' dataset specifically
            y_data = source_file['Y'][:NUM_SAMPLES]
            y_sample = y_data
            sample_file.create_dataset('Y', data=y_sample)

            # --- Process Z dataset (SNRs) ---
            print(f"Slicing first {NUM_SAMPLES} records from /Z...")
            z_data = source_file['Z'][:NUM_SAMPLES]
            z_sample = z_data
            sample_file.create_dataset('Z', data=z_sample)

            print("\nSample file created successfully!")
            print(f"Location: {SAMPLE_FILE_PATH}")
            print(f"Contains {len(x_sample)} records.")

    except FileNotFoundError:
        print(f"ERROR: Source file not found at {SOURCE_FILE_PATH}")
        print("Please ensure the full dataset is in the 'data' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    create_data_sample()
