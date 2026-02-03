"""
Quick test: Load existing HW images from disk, train in-memory like original working script.
This tests if the images are learnable, using the exact same approach that got 57%.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from tqdm import tqdm

print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# =============================================================================
# MODEL (exact copy from working phase2_channel_comparison.py)
# =============================================================================
def build_squeezenet_simple(input_shape, num_classes):
    """Exact model from working original."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    def fire_module(x, squeeze, expand):
        squeeze_out = layers.Conv2D(squeeze, (1, 1), activation='relu', padding='same')(x)
        expand_1x1 = layers.Conv2D(expand, (1, 1), activation='relu', padding='same')(squeeze_out)
        expand_3x3 = layers.Conv2D(expand, (3, 3), activation='relu', padding='same')(squeeze_out)
        return layers.Concatenate()([expand_1x1, expand_3x3])
    
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = fire_module(x, 32, 128)
    x = fire_module(x, 32, 128)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = fire_module(x, 64, 256)
    
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax')(x)
    
    return keras.Model(inputs, outputs)


# =============================================================================
# LOAD IMAGES INTO MEMORY (like original did)
# =============================================================================
def load_images_from_disk(data_dir, max_per_class=2000):
    """Load images into memory, normalize to [0,1] like original."""
    images = []
    labels = []
    
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_dirs)}
    
    print(f"Classes: {class_dirs}")
    
    for class_name in tqdm(class_dirs, desc="Loading classes"):
        class_path = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith('.png')][:max_per_class]
        
        for fname in files:
            img = Image.open(os.path.join(class_path, fname))
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize like original!
            images.append(img_array)
            labels.append(class_to_idx[class_name])
    
    return np.array(images), np.array(labels), class_dirs


# =============================================================================
# MAIN
# =============================================================================
def main():
    train_dir = 'data/processed_hw_single/train'
    val_dir = 'data/processed_hw_single/validation'
    
    if not os.path.exists(train_dir):
        print(f"ERROR: {train_dir} does not exist!")
        print("Run the full script first to generate images.")
        return
    
    print("\n" + "="*60)
    print("QUICK HW IMAGE TRAINING TEST")
    print("Using exact same approach as working 57% script")
    print("="*60)
    
    # Load into memory (limit samples for quick test)
    print("\nLoading training images...")
    X_train, y_train, class_names = load_images_from_disk(train_dir, max_per_class=2000)
    print(f"Train: {X_train.shape}, labels: {y_train.shape}")
    
    print("\nLoading validation images...")
    X_val, y_val, _ = load_images_from_disk(val_dir, max_per_class=500)
    print(f"Val: {X_val.shape}, labels: {y_val.shape}")
    
    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    # Build model
    print(f"\nInput shape: {X_train.shape[1:]}")
    model = build_squeezenet_simple(X_train.shape[1:], len(class_names))
    
    # Compile with RMSprop like original
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model params: {model.count_params():,}")
    
    # Train for 10 epochs
    print("\n" + "="*60)
    print("TRAINING (10 epochs, batch_size=32)")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n>>> Final Val Accuracy: {final_val_acc*100:.2f}%")
    
    if final_val_acc > 0.20:
        print("SUCCESS! Images are learnable. The issue was with disk loading pipeline.")
    else:
        print("STILL STUCK. Issue might be with the images themselves.")


if __name__ == "__main__":
    main()
