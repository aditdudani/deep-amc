import numpy as np
from data_loader import load_data_sample
import matplotlib.pyplot as plt
import tensorflow as tf

def _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alpha, plane_range=7.0):
    """
    Converts a batch of I/Q samples to a single enhanced gray image using vectorized TensorFlow ops.
    This function is a corrected implementation that aligns with the Peng et al. (2018) methodology.
    
    Args:
        iq_samples (tf.Tensor): A tensor of I/Q samples with shape [num_samples, 2].
        grid_size (int): The resolution of the output image (e.g., 224 for a 224x224 image).
        alpha (float): The exponential decay rate parameter.
        plane_range (float): The symmetric range of the complex plane (e.g., 7.0 for a -3.5 to 3.5 grid).

    Returns:
        tf.Tensor: A tensor representing the single-channel grayscale image with shape [grid_size, grid_size].
    """
    # 1. CORRECTION: Define the coordinate grid using the specified plane_range.
    # This creates the correct [-3.5, 3.5] canvas instead of [-1.0, 1.0].
    coords = tf.linspace(-plane_range / 2.0, plane_range / 2.0, grid_size)
    grid_x, grid_y = tf.meshgrid(coords, coords)
    pixel_centers = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=1) # Shape: [grid_size*grid_size, 2]

    # Prepare tensors for broadcasting
    # iq_samples shape:      [num_samples, 2] -> [num_samples, 1, 2]
    # pixel_centers shape: [grid_size*grid_size, 2] -> [1, grid_size*grid_size, 2]
    iq_samples_b = iq_samples[:, tf.newaxis, :]
    pixel_centers_b = pixel_centers[tf.newaxis, :, :]

    # Calculate squared Euclidean distance
    dist_sq = tf.reduce_sum(tf.square(iq_samples_b - pixel_centers_b), axis=2) # Shape: [num_samples, grid_size*grid_size]

    # 2. CORRECTION: Apply tf.sqrt to get the true Euclidean distance.
    # This fixes the mathematical error of using a Gaussian instead of an exponential decay.
    distances = tf.sqrt(dist_sq)

    # Calculate influences based on the corrected distance
    influences = tf.exp(-alpha * distances) # P=1 is assumed

    # Sum influences for each pixel
    pixel_intensities = tf.reduce_sum(influences, axis=0)
    
    # Reshape the 1D intensity vector back into a 2D image
    image = tf.reshape(pixel_intensities, (grid_size, grid_size))

    # 3. CORRECTION: Normalize by dividing by the maximum value to match the NumPy baseline.
    # This replaces the incorrect min-max scaling.
    image_max = tf.reduce_max(image)
    if image_max > 0:
        image = image / image_max
    
    return image

def tf_generate_three_channel_image(iq_samples, grid_size=224, alphas=(10.0, 1.0, 0.1), plane_range=7.0):
    """
    Generates a 3-channel image from I/Q samples by stacking three corrected enhanced gray images
    with different alpha values. Implemented with pure TensorFlow operations.

    Args:
        iq_samples (tf.Tensor): A tensor of I/Q samples with shape [num_samples, 2].
        grid_size (int): The resolution of the output image.
        alphas (tuple of float): A tuple of three alpha values for the decay function.
        plane_range (float): The symmetric range of the complex plane.

    Returns:
        tf.Tensor: A 3-channel image tensor with shape [grid_size, grid_size, 3].
    """
    # Generate each channel using the corrected vectorized gray image function
    image_ch1 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[0], plane_range)
    image_ch2 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[1], plane_range)
    image_ch3 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[2], plane_range)

    # Stack the channels to form the final 3-channel image
    three_channel_image = tf.stack([image_ch1, image_ch2, image_ch3], axis=-1)
    
    return three_channel_image

if __name__ == '__main__':
    print("Testing image generator...")
    sample_path = '~/amc_project/data/RML2018.01A_sample.h5'
    X_sample, _, _ = load_data_sample(sample_path)
    
    n=1
    signal_frame = X_sample[n] # Pick just the nth frame
    print(f"Processing {n}th signal frame with shape: {signal_frame.shape}")
    
    generated_image = tf_generate_three_channel_image(signal_frame)
    
    print(f"Generated image shape: {generated_image.shape}")    

    print("Displaying the 3 channels of the generated image separately...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alphas_for_title = (10, 1, 0.1)
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(generated_image[:, :, i], cmap='viridis')
        ax.set_title(f'Channel {i+1} (alpha={alphas_for_title[i]})')
        fig.colorbar(im, ax=ax)     
    plt.suptitle("Individual Channels of the Generated Image")
    
    print("\nDisplaying the combined 3-channel image...")
    plt.figure(figsize=(8, 8))
    plt.imshow(generated_image)
    plt.title(f"Combined 3-Channel Image (Frame {n})")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    
    plt.show()