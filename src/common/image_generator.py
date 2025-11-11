from common.data_loader import load_data_sample
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alpha, plane_range=7.0):
    coords = tf.linspace(-plane_range / 2.0, plane_range / 2.0, grid_size)
    grid_x, grid_y = tf.meshgrid(coords, coords)
    pixel_centers = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=1)
    iq_samples_b = iq_samples[:, tf.newaxis, :]
    pixel_centers_b = pixel_centers[tf.newaxis, :, :]
    dist_sq = tf.reduce_sum(tf.square(iq_samples_b - pixel_centers_b), axis=2)
    distances = tf.sqrt(dist_sq)
    influences = tf.exp(-alpha * distances)
    pixel_intensities = tf.reduce_sum(influences, axis=0)
    image = tf.reshape(pixel_intensities, (grid_size, grid_size))
    image_max = tf.reduce_max(image)
    if image_max > 0:
        image = image / image_max
    return image

def tf_generate_three_channel_image(iq_samples, grid_size=224, alphas=(10.0, 1.0, 0.1), plane_range=7.0):
    image_ch1 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[0], plane_range)
    image_ch2 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[1], plane_range)
    image_ch3 = _tf_iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[2], plane_range)
    three_channel_image = tf.stack([image_ch1, image_ch2, image_ch3], axis=-1)
    return three_channel_image

if __name__ == '__main__':
    print("Testing image generator (common)...")
    sample_path = '~/amc_project/data/RML2018.01A_sample.h5'
    X_sample, _, _ = load_data_sample(sample_path)
    signal_frame = X_sample[0]
    generated_image = tf_generate_three_channel_image(signal_frame)
    print(f"Generated image shape: {generated_image.shape}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alphas_for_title = (10, 1, 0.1)
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(generated_image[:, :, i], cmap='viridis')
        ax.set_title(f'Channel {i+1} (alpha={alphas_for_title[i]})')
        fig.colorbar(im, ax=ax)
    plt.show()
