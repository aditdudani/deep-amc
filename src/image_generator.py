# src/image_generator.py
import numpy as np

def iq_to_binary_constellation(iq_samples, resolution=224):
    """
    Converts a single I/Q sample frame to a 2D binary constellation image.

    Args:
        iq_samples (np.array): A single I/Q sample frame of shape (2, N) or (N, 2).
        resolution (int): The width and height of the output square image.

    Returns:
        np.array: A 2D binary image of shape (resolution, resolution).
    """
    # Ensure I/Q samples are in the correct format (2, N)
    if iq_samples.shape!= 2:
        iq_samples = iq_samples.T

    I = iq_samples[0, :]  # In-phase component
    Q = iq_samples[1, :]  # Quadrature component
    
    # Create a 2D histogram. This counts the number of points in each pixel bin.
    # The range is set to [-2, 2] which is a reasonable assumption for normalized signals.
    hist, _, _ = np.histogram2d(I, Q, bins=resolution, range=[[-2, 2], [-2, 2]])
    
    # Binarize the image: a pixel is 1 if it contains one or more points, 0 otherwise.
    binary_image = (hist > 0).astype(np.float32)
    
    # Transpose for correct image orientation (imshow expects rows, columns).
    return np.transpose(binary_image)

def iq_to_3channel_binary(iq_samples, resolution=224):
    """
    Creates a 3-channel image by stacking a binary constellation. This is necessary
    to feed the image into pre-trained models that expect an RGB input.

    Args:
        iq_samples (np.array): A single I/Q sample frame.
        resolution (int): The width and height of the output square image.

    Returns:
        np.array: A 3-channel binary image of shape (resolution, resolution, 3).
    """
    binary_image = iq_to_binary_constellation(iq_samples, resolution)
    
    # Use np.stack to repeat the single-channel image three times along a new last axis.
    return np.stack([binary_image, binary_image, binary_image], axis=-1)