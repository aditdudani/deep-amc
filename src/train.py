import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
# Tqdm is no longer needed as Keras provides the progress bar
# from tqdm import tqdm 

from data_loader import load_data_sample
# Import the NEW TensorFlow wrapper function
from image_generator import tf_generate_three_channel_image 
from model_builder import build_googlenet_transfer

# --- Configuration ---
SAMPLE_PATH = '/app/data/RML2018.01A_sample_1.h5'
NUM_CLASSES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# The old data_generator function is no longer needed and can be deleted.

def main():
    print("--- Starting Training Pipeline ---")

    # --- Step 1: Load Data ---
    print("Step 1: Loading data sample...")
    X_iq, Y_onehot, Z_snr = load_data_sample(SAMPLE_PATH)
    Y_labels = np.argmax(Y_onehot, axis=1)
    print(f"Loaded {len(X_iq)} total samples.")

    # --- Step 2: Split Data (This part is already correct) ---
    print("Step 2: Splitting data into training and validation sets...")
    X_iq_train, X_iq_val, Y_labels_train, Y_labels_val = train_test_split(
        X_iq,
        Y_labels,
        test_size=0.2,
        random_state=42,
        stratify=Y_labels
    )
    print(f"Training set size: {len(X_iq_train)}")
    print(f"Validation set size: {len(X_iq_val)}")

    # --- Step 3: Create TensorFlow Data Pipelines (The Optimized Way) ---
    print("Step 3: Creating tf.data pipelines...")

    # Create the training dataset from the raw I/Q data
    train_dataset = tf.data.Dataset.from_tensor_slices((X_iq_train, Y_labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024) # Shuffle for better training
    # Use.map() to apply the image generation in parallel
    train_dataset = train_dataset.map(
        lambda x, y: (tf_generate_three_channel_image(x), y), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) # Prefetch for performance

    # Create the validation dataset pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((X_iq_val, Y_labels_val))
    val_dataset = val_dataset.map(
        lambda x, y: (tf_generate_three_channel_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # --- Step 4: Build and Compile the Model (No changes needed here) ---
    print("Step 4: Building the GoogLeNet model...")
    model = build_googlenet_transfer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # --- Step 5: Train the Model using the Data Pipeline (No changes needed here) ---
    print("\nStep 5: Starting model training...")
    os.makedirs('models', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath='models/best_model_local.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # The verbose=1 argument for model.fit is now needed to see the progress bar
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback],
        verbose=1 
    )

    print("\n--- Training finished successfully! ---")

if __name__ == '__main__':
    main()