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
    
    # DIAGNOSTIC: Check image statistics
    print("\n" + "="*60)
    print("IMAGE STATISTICS")
    print("="*60)
    print(f"  Overall - mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
    for i, cls in enumerate(class_names):
        cls_mask = y_train == i
        cls_mean = X_train[cls_mask].mean()
        cls_std = X_train[cls_mask].std()
        print(f"  {cls:8s} - mean: {cls_mean:.4f}, std: {cls_std:.4f}")
    
    # Build model
    print(f"\nInput shape: {X_train.shape[1:]}")
    
    print(f"Testing multiple optimizers to find one that works...")
    
    # TRY MULTIPLE OPTIMIZERS to find one that breaks out of saddle point
    optimizers_to_try = [
        ('Adam_lr0.001', lambda: keras.optimizers.Adam(learning_rate=0.001)),
        ('Adam_lr0.01', lambda: keras.optimizers.Adam(learning_rate=0.01)),
        ('SGD_lr0.01_mom0.9', lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
        ('SGD_lr0.1_mom0.9', lambda: keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ]
    
    for opt_name, opt_fn in optimizers_to_try:
        print("\n" + "="*60)
        print(f"TESTING: {opt_name}")
        print("="*60)
        
        # Rebuild model fresh for each optimizer
        tf.keras.backend.clear_session()
        model = build_squeezenet_simple(X_train.shape[1:], len(class_names))
        model.compile(
            optimizer=opt_fn(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model params: {model.count_params():,}")
        
        # Train for just 5 epochs to check if it breaks out
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            verbose=1
        )
        
        final_val_acc = history.history['val_accuracy'][-1]
        print(f">>> {opt_name} Val Accuracy after 5 epochs: {final_val_acc*100:.2f}%")
        
        if final_val_acc > 0.15:
            print(f"SUCCESS! {opt_name} broke out of saddle point!")
            # Continue training with this optimizer
            print("\nContinuing training for 15 more epochs...")
            history2 = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=15,
                batch_size=32,
                verbose=1
            )
            final_val_acc = history2.history['val_accuracy'][-1]
            print(f">>> After 20 total epochs: {final_val_acc*100:.2f}%")
            break
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
