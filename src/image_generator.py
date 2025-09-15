import numpy as np
from data_loader import load_data_sample
import matplotlib.pyplot as plt

def iq_to_enhanced_gray_image(iq_samples, grid_size, alpha, plane_range=7.0):
    image = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    #creating empty grid
    pixel_coords = np.linspace(-plane_range / 2, plane_range / 2, grid_size)
    #print(f"Pixel coords shape: {pixel_coords.shape}")
    grid_x, grid_y = np.meshgrid(pixel_coords, pixel_coords)
    pixel_centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    #print(f"Pixel centers shape: {pixel_centers.shape}")
    #calculates distance from every sample to every pixel center
    #iq_samples shape:    (1024, 2)  -> (1024, 1, 2)
    #pixel_centers shape: (50176, 2) -> (1, 50176, 2)
    distances = np.sqrt(np.sum((iq_samples[:, np.newaxis, :] - pixel_centers[np.newaxis, :, :])**2, axis=2))
    influences = np.exp(-alpha * distances) #assume P=1
    pixel_intensities = np.sum(influences, axis=0) #sum influences for each pixel

    image = pixel_intensities.reshape((grid_size, grid_size)) #now image is 2D grid
    
    if image.max() > 0:
        image /= image.max() #normalize
        
    return image

def generate_three_channel_image(iq_samples, grid_size=224, alphas=(0.1, 1.0, 10.0)): 

    image_ch1 = iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[0])
    image_ch2 = iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[1])
    image_ch3 = iq_to_enhanced_gray_image(iq_samples, grid_size, alphas[2])

    three_channel_image = np.stack([image_ch1, image_ch2, image_ch3], axis=-1)
    
    return three_channel_image

if __name__ == '__main__':
    print("Testing image generator...")
    sample_path = '~/amc_project/data/RML2018.01A_sample.h5'
    X_sample, _, _ = load_data_sample(sample_path)
    
    n=0
    signal_frame = X_sample[n] #pick just the nth frame
    print(f"Processing {n}th signal frame with shape: {signal_frame.shape}")
    
    generated_image = generate_three_channel_image(signal_frame)
    
    print(f"Generated image shape: {generated_image.shape}")    

    print("Displaying the 3 channels of the generated image...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alphas_for_title = (0.1, 1.0, 10.0)
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(generated_image[:, :, i], cmap='viridis')
        ax.set_title(f'Channel {i+1} (alpha={alphas_for_title[i]})')
        fig.colorbar(im, ax=ax)
        
    plt.suptitle("Generated 3-Channel Image from First Data Sample")
    plt.show()