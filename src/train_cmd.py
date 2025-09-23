import os
import argparse # Import the argparse library
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from data_loader import load_data_sample
from image_generator import generate_three_channel_image
from model_builder import build_googlenet_transfer

# --- Configuration is now handled by argparse in main() ---

def data_generator(X_iq, Y_labels):
    """A generator function that yields one image and its label at a time."""
    for i in range(len(X_iq)):
        image = generate_three_channel_image(X_iq[i])
        image = image.astype(np.float32)
        label = np.int64(Y_labels[i])
        yield image, label

def main(args): # The main function now accepts an 'args' object
    print("--- Starting Training Pipeline ---")

    # Expand ~ in paths for robustness
    data_path = os.path.expanduser(args.data_path)
    model_save_path = os.path.expanduser(args.model_save_path)

    # --- Step 1: Load Data ---
    print(f"Step 1: Loading data from {data_path}...")
    X_iq, Y_onehot, Z_snr = load_data_sample(data_path)
    Y_labels = np.argmax(Y_onehot, axis=1)
    print(f"Loaded {len(X_iq)} total samples.")

    # --- Step 2: Split Data Indices ---
    print("Step 2: Splitting data indices...")
    indices = np.arange(len(X_iq))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=Y_labels
    )
    X_iq_train, Y_labels_train = X_iq[train_indices], Y_labels[train_indices]
    X_iq_val, Y_labels_val = X_iq[val_indices], Y_labels[val_indices]
    print(f"Training set size: {len(X_iq_train)}")
    print(f"Validation set size: {len(X_iq_val)}")

    # --- Step 3: Create TensorFlow Data Pipelines ---
    print("Step 3: Creating tf.data pipelines...")
    output_signature = (
        tf.TensorSpec(shape=(args.image_size, args.image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_iq_train, Y_labels_train), output_signature=output_signature
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_iq_val, Y_labels_val), output_signature=output_signature
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # --- Step 4: Build and Compile the Model ---
    print("Step 4: Building the GoogLeNet model...")
    model = build_googlenet_transfer(
        input_shape=(args.image_size, args.image_size, 3), num_classes=args.num_classes
    )
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # --- Step 5: Train the Model ---
    print("\nStep 5: Starting model training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
    )
    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback]
    )
    print(f"\n--- Training finished successfully! Best model saved to {model_save_path} ---")

if __name__ == '__main__':
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description='AMC Model Training Script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the H5 dataset file.')
    parser.add_argument('--model_save_path', type=str, default='models/googlenet_baseline.h5', help='Path to save the best model.')
    parser.add_argument('--num_classes', type=int, default=24, help='Number of modulation classes.')
    parser.add_argument('--image_size', type=int, default=224, help='Resolution of the generated images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    
    args = parser.parse_args()
    main(args)