import numpy as np
from data_loader import load_data_sample
import matplotlib.pyplot as plt
import tensorflow as tf

def iq_to_enhanced_gray_image(iq_samples, grid_size, alpha, plane_range=7.0):
    """
    Generates an enhanced gray image using a memory-efficient iterative approach.
    """
    image = np.zeros((grid_size, grid_size), dtype=np.float32)
    pixel_coords = np.linspace(-plane_range / 2, plane_range / 2, grid_size)
    
    # Iterate through each I/Q sample to avoid creating a massive intermediate array
    for sample in iq_samples:
        # Calculate distance from this single sample to all pixel centers
        dist_sq = (pixel_coords - sample)**2 + (pixel_coords[:, np.newaxis] - sample[1])**2
        
        # Calculate influence and add it to the image grid
        influence = np.exp(-alpha * np.sqrt(dist_sq))
        image += influence

    if image.max() > 0:
        image /= image.max()
        
    return image

def generate_three_channel_image(iq_samples, grid_size=224, alphas=(10, 1, 0.1)): 

    image_ch1 = iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[0])
    image_ch2 = iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[1])
    image_ch3 = iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[2])

    three_channel_image = np.stack([image_ch1, image_ch2, image_ch3], axis=-1)
    
    return three_channel_image

# --- NEW PART: Add a TensorFlow wrapper for the pipeline ---
@tf.function
def tf_generate_three_channel_image(iq_samples):
  # Use tf.py_function to wrap the NumPy-based function
  # This tells TensorFlow how to execute your Python code in its graph
  [image,] = tf.py_function(generate_three_channel_image, [iq_samples], [tf.float32])
  
  # Set the shape explicitly so TensorFlow knows what to expect
  image.set_shape((224, 224, 3))
  return image

if __name__ == '__main__':
    print("Testing image generator...")
    sample_path = '~/amc_project/data/RML2018.01A_sample_1.h5'
    X_sample, _, _ = load_data_sample(sample_path)
    
    n=5000
    signal_frame = X_sample[n] # Pick just the nth frame
    print(f"Processing {n}th signal frame with shape: {signal_frame.shape}")
    
    generated_image = generate_three_channel_image(signal_frame)
    
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