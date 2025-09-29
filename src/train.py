import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision

# --- Local Module Imports ---
from data_loader import load_data_sample
from image_generator import tf_generate_three_channel_image
from model_builder import build_googlenet_transfer

# --- 1. V100 Performance Optimization: Enable Mixed-Precision Training ---
# This is the critical step to activate the V100's Tensor Cores.
# It instructs Keras to use 16-bit precision for computations where possible,
# and 32-bit for variable storage, dramatically accelerating training.
mixed_precision.set_global_policy('mixed_float16')
print("Mixed-precision training policy set to 'mixed_float16'.")

# --- 2. Configuration ---
SAMPLE_PATH = '/app/data/RML2018.01A_sample_1.h5'
MODEL_SAVE_PATH = 'models/googlenet_phase1_best.h5'
NUM_CLASSES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20 # Increased for a more robust training session

def main():
    """Main function to orchestrate the training pipeline."""
    print("\n--- Starting AMC Training Pipeline ---")

    # --- 3. Load and Prepare Data ---
    print("\nStep 1: Loading data sample...")
    X_iq, Y_onehot, _ = load_data_sample(SAMPLE_PATH)
    
    # Convert one-hot encoded labels to integer labels for SparseCategoricalCrossentropy
    Y_labels = np.argmax(Y_onehot, axis=1)
    print(f"Loaded {len(X_iq)} total samples.")

    # --- 4. Split Data ---
    print("\nStep 2: Splitting data into training and validation sets...")
    X_iq_train, X_iq_val, Y_labels_train, Y_labels_val = train_test_split(
        X_iq,
        Y_labels,
        test_size=0.2,
        random_state=42,
        stratify=Y_labels  # Ensure class distribution is the same in train/val sets
    )
    print(f"Training set size: {len(X_iq_train)}")
    print(f"Validation set size: {len(X_iq_val)}")

    # --- 5. Create High-Performance tf.data Pipelines ---
    print("\nStep 3: Creating optimized tf.data pipelines...")

    def process_sample(iq_samples, label):
        """
        Applies the TensorFlow-native image generation function.
        This function will be traced by tf.data.map() into a static graph.
        """
        # Directly call the fully vectorized, TF-native image generator.
        # No tf.py_function is needed, ensuring maximum performance.
        image = tf_generate_three_channel_image(iq_samples, grid_size=IMAGE_SIZE)
        # Explicitly set the shape. This is crucial for graph optimization.
        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return image, label

    # Create the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_iq_train, Y_labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024) # Shuffle for better training
    train_dataset = train_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) # Overlap data preprocessing and model execution

    # Create the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_iq_val, Y_labels_val))
    val_dataset = val_dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    print("Data pipelines created successfully.")

    # --- 6. Build and Compile the Model ---
    print("\nStep 4: Building and compiling the GoogLeNet model...")
    model = build_googlenet_transfer(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 
        num_classes=NUM_CLASSES
    )
    
    # Create the base optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Wrap the optimizer in LossScaleOptimizer for mixed-precision training.
    # This automatically handles loss scaling to prevent numerical underflow.
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()
    print("Model compilation complete.")

    # --- 7. Train the Model ---
    print("\nStep 5: Starting model training...")
    
    # Ensure the directory for saving models exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Create a callback to save only the best model during training
    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    # Train the model using the efficient data pipelines
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