import os
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision

# --- Local Module Imports ---
from image_generator import tf_generate_three_channel_image
from model_builder import build_googlenet_transfer

# --- 1. V100 Performance Optimization: Enable Mixed-Precision Training ---
mixed_precision.set_global_policy('mixed_float16')
print("Mixed-precision training policy set to 'mixed_float16'.")

# --- 2. Configuration ---
# IMPORTANT: Update this path to point to your full dataset file
FULL_DATA_PATH = '/app/data/GOLD_XYZ_OSC.0001_1024.hdf5' 
MODEL_SAVE_PATH = 'models/googlenet_full_dataset.h5'
NUM_CLASSES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 64 # Increased for the V100
EPOCHS = 20

# --- 3. High-Performance Data Generator ---
def hdf5_generator(h5_path, indices):
    """
    A Python generator that yields data samples and labels directly from an HDF5 file.
    This avoids loading the entire dataset into RAM.
    """
    def gen():
        with h5py.File(h5_path, 'r') as hf:
            X = hf['X']
            Y = hf['Y']
            for i in indices:
                iq_sample = X[i].astype(np.float32)
                label = np.argmax(Y[i])
                yield iq_sample, label
    return gen

def main():
    """Main function to orchestrate the training pipeline."""
    print("\n--- Starting AMC Training Pipeline with Full Dataset ---")

    # --- 4. Load Data Indices (Not the Data Itself) ---
    print("\nStep 1: Loading data indices and labels for splitting...")
    with h5py.File(FULL_DATA_PATH, 'r') as hf:
        num_samples = hf['X'].shape[0]
        # Load only the labels for stratification, which is memory-efficient
        all_labels = np.argmax(hf['Y'][:], axis=1)

    indices = np.arange(num_samples)
    print(f"Found {num_samples} total samples in the dataset.")

    # --- 5. Split Indices ---
    print("\nStep 2: Splitting data indices into training and validation sets...")
    # We split the indices, not the actual data. This is extremely fast and memory-light.
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")

    # --- 6. Create High-Performance tf.data Pipelines from Generator ---
    print("\nStep 3: Creating optimized tf.data pipelines from generator...")

    # Define the data types and shapes that our generator will produce.
    # This is crucial for TensorFlow to build the computation graph correctly.
    output_signature = (
        tf.TensorSpec(shape=(1024, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    # Create the training dataset from our HDF5 generator
    train_dataset = tf.data.Dataset.from_generator(
        hdf5_generator(FULL_DATA_PATH, train_indices),
        output_signature=output_signature
    )

    # Create the validation dataset from our HDF5 generator
    val_dataset = tf.data.Dataset.from_generator(
        hdf5_generator(FULL_DATA_PATH, val_indices),
        output_signature=output_signature
    )

    def process_sample(iq_samples, label):
        """Applies the on-the-fly image generation function."""
        image = tf_generate_three_channel_image(iq_samples, grid_size=IMAGE_SIZE)
        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return image, label

    # Apply the standard high-performance transformations
    train_dataset = train_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    print("Data pipelines created successfully.")

    # --- 7. Build and Compile the Model (No changes needed here) ---
    print("\nStep 4: Building and compiling the GoogLeNet model...")
    model = build_googlenet_transfer(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 
        num_classes=NUM_CLASSES
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    #model.summary()
    print("Model compilation complete.")

    # --- 8. Train the Model ---
    print("\nStep 5: Starting model training...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback],
        verbose=1
    )

    print(f"\n--- Training finished successfully! Best model saved to {MODEL_SAVE_PATH} ---")

if __name__ == '__main__':
    main()
