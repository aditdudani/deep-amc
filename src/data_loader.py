import pickle
import numpy as np

def load_data(filename):
    """
    Loads the RadioML dataset from a pickle file.
    
    The dataset is expected to be a dictionary where keys are tuples of 
    (modulation, snr) and values are numpy arrays of IQ samples.
    
    Args:
        filename (str): The full path to the .pkl dataset file.
        
    Returns:
        dict: The loaded dataset.
    """
    # The original RadioML datasets were created with Python 2.
    # [cite_start]The 'latin1' encoding is crucial for unpickling them in Python 3. [cite: 736]
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def filter_by_mods_and_snrs(data, target_mods, target_snrs=None):
    """
    Filters the dataset to include only specified modulations and SNRs.

    Args:
        data (dict): The full dataset loaded by load_data.
        target_mods (list): A list of strings of modulation types to include.
        target_snrs (list, optional): A list of integers of SNR values to include. 
                                     If None, all SNRs are included. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The filtered IQ data samples (X).
            - np.ndarray: The corresponding integer labels (y).
            - np.ndarray: The corresponding SNR values for each sample.
    """
    X = []
    y = []
    snrs = []
    
    # Create a mapping from modulation name to an integer label
    mod_to_int = {mod: i for i, mod in enumerate(target_mods)}

    for mod_snr_tuple, samples in data.items():
        mod, snr = mod_snr_tuple
        
        # Check if the current modulation and SNR are in the target lists
        is_mod_targeted = mod in target_mods
        is_snr_targeted = (target_snrs is None) or (snr in target_snrs)
        
        if is_mod_targeted and is_snr_targeted:
            X.extend(samples)
            # Append the integer label for the current modulation
            y.extend([mod_to_int[mod]] * len(samples))
            snrs.extend([snr] * len(samples))
            
    return np.array(X), np.array(y), np.array(snrs)