
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from data_loader import load_data_sample
from image_generator import generate_three_channel_image
from model_builder import build_googlenet_transfer

# --- Configuration ---
SAMPLE_PATH = '/app/projects/amc_project/data/RML2018.01A_sample_1.h5'
NUM_CLASSES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 32 # Define batch size for the data pipeline
EPOCHS = 10

def data_generator(X_iq, Y_labels):
    """A generator function that yields one image and its label at a time."""
    for i in range(len(X_iq)):
        image = generate_three_channel_image(X_iq[i])
        # Ensure image is float32 and correct shape
        image = image.astype(np.float32)
        label = np.int64(Y_labels[i])
        yield image, label

def main():
    print("--- Starting Training Pipeline ---")

    # --- Step 1: Load Data ---
    print("Step 1: Loading data sample...")
    X_iq, Y_onehot, Z_snr = load_data_sample(SAMPLE_PATH)
    Y_labels = np.argmax(Y_onehot, axis=1) # Convert one-hot to integer labels
    print(f"Loaded {len(X_iq)} total samples.")

    # --- Step 2: Split Data Indices (Not the data itself) ---
    print("Step 2: Splitting data indices into training and validation sets...")
    indices = np.arange(len(X_iq))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=Y_labels
    )

    X_iq_train, Y_labels_train = X_iq[train_indices], Y_labels[train_indices]
    X_iq_val, Y_labels_val = X_iq[val_indices], Y_labels[val_indices]

    print(f"Training set size: {len(X_iq_train)}")
    print(f"Validation set size: {len(X_iq_val)}")

    # --- Step 3: Create TensorFlow Data Pipelines ---
    print("Step 3: Creating tf.data pipelines...")

    # Define the output signature for the generator
    output_signature = (
        tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    # Create the training dataset pipeline
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_iq_train, Y_labels_train),
        output_signature=output_signature
    )
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create the validation dataset pipeline
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_iq_val, Y_labels_val),
        output_signature=output_signature
    )
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Step 4: Build and Compile the Model ---
    print("Step 4: Building the GoogLeNet model...")
    model = build_googlenet_transfer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # --- Step 5: Train the Model using the Data Pipeline ---
    print("\nStep 5: Starting model training...")

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath='models/best_model_local.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Pass the tf.data.Dataset objects directly to model.fit
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback]
    )

    print("\n--- Training finished successfully! ---")

if __name__ == '__main__':
    main()