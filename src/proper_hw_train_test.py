"""
Proper HW image training test - matches successful adaptive sampling setup.
Uses full training regime, not short bursts.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from PIL import Image
from tqdm import tqdm

# Force GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

print(f"GPUs: {len(gpus)}")

# =============================================================================
# MODEL - Using common/squeezenet.py style (CORRECT - no activation before softmax)
# =============================================================================
def _fire_module(x, squeeze_channels, expand_channels):
    squeeze = layers.Conv2D(squeeze_channels, (1, 1), activation='relu', padding='valid',
                            kernel_initializer='he_normal')(x)
    expand_1x1 = layers.Conv2D(expand_channels, (1, 1), activation='relu', padding='valid',
                               kernel_initializer='he_normal')(squeeze)
    expand_3x3 = layers.Conv2D(expand_channels, (3, 3), activation='relu', padding='same',
                               kernel_initializer='he_normal')(squeeze)
    return layers.Concatenate(axis=-1)([expand_1x1, expand_3x3])


def build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=8, dropout_rate=0.0):
    """SqueezeNet v1.1 - EXACT copy from common/squeezenet.py"""
    inputs = layers.Input(shape=input_shape, dtype='float32')
    x = layers.Rescaling(1.0 / 255.0, name='rescale_0_1')(inputs)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', 
                      name='conv1', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)
    x = _fire_module(x, squeeze_channels=16, expand_channels=64)
    x = _fire_module(x, squeeze_channels=16, expand_channels=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool3')(x)
    x = _fire_module(x, squeeze_channels=32, expand_channels=128)
    x = _fire_module(x, squeeze_channels=32, expand_channels=128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool5')(x)
    x = _fire_module(x, squeeze_channels=48, expand_channels=192)
    x = _fire_module(x, squeeze_channels=48, expand_channels=192)
    x = _fire_module(x, squeeze_channels=64, expand_channels=256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool8')(x)
    x = _fire_module(x, squeeze_channels=64, expand_channels=256)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='dropout')(x)
    # CRITICAL: activation=None here, softmax is separate
    x = layers.Conv2D(num_classes, (1, 1), activation=None, padding='valid', 
                      name='conv_final', dtype='float32', kernel_initializer='he_normal')(x)
    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    outputs = layers.Softmax(name='predictions')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='squeezenet_v1_1')


# =============================================================================
# LOAD IMAGES - DON'T normalize (model has Rescaling layer)
# =============================================================================
def load_images_from_disk(data_dir, max_per_class=None):
    """Load images as uint8 (0-255), model will normalize."""
    images = []
    labels = []
    
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_dirs)}
    
    print(f"Classes: {class_dirs}")
    
    for class_name in tqdm(class_dirs, desc="Loading"):
        class_path = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        if max_per_class:
            files = files[:max_per_class]
        
        for fname in files:
            img = Image.open(os.path.join(class_path, fname))
            img_array = np.array(img, dtype=np.float32)  # Keep as 0-255, model normalizes
            images.append(img_array)
            labels.append(class_to_idx[class_name])
    
    return np.array(images), np.array(labels), class_dirs


# =============================================================================
# MAIN - Proper training like adaptive_sampling/train_squeezenet_sampler.py
# =============================================================================
def main():
    train_dir = 'data/processed_hw_single/train'
    val_dir = 'data/processed_hw_single/validation'
    
    if not os.path.exists(train_dir):
        print(f"ERROR: {train_dir} does not exist!")
        return
    
    print("\n" + "="*70)
    print("PROPER HW IMAGE TRAINING TEST")
    print("Matching adaptive_sampling/train_squeezenet_sampler.py settings")
    print("="*70)
    
    # Training config - MATCHING your working setup
    EPOCHS = 40
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2  # Your working setup used 1e-2
    
    print(f"\nConfig: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("Optimizer: SGD with momentum=0.9 (matching your working setup)")
    
    # Load images
    print("\nLoading training images...")
    X_train, y_train, class_names = load_images_from_disk(train_dir, max_per_class=5000)
    print(f"Train: {X_train.shape}")
    
    print("\nLoading validation images...")
    X_val, y_val, _ = load_images_from_disk(val_dir, max_per_class=None)
    print(f"Val: {X_val.shape}")
    
    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    # Build model - EXACT architecture from common/squeezenet.py
    print(f"\nBuilding model...")
    model = build_squeezenet_v11(
        input_shape=(224, 224, 3),
        num_classes=len(class_names),
        dropout_rate=0.0  # Your working setup used 0.0
    )
    
    # Optimizer: SGD with momentum - MATCHING your working setup
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    print(f"Model params: {model.count_params():,}")
    
    # Callbacks - MATCHING your working setup
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        verbose=1
    )
    
    class LrPrinter(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch % 10 == 0:
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                print(f"\n[Epoch {epoch+1}] LR: {lr:.6g}")
    
    # Train
    print("\n" + "="*70)
    print(f"TRAINING FOR {EPOCHS} EPOCHS")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[reduce_lr, LrPrinter()],
        verbose=1
    )
    
    # Results
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Final Val Accuracy: {final_val_acc*100:.2f}%")
    print(f"  Best Val Accuracy:  {best_val_acc*100:.2f}% (epoch {best_epoch})")
    
    if best_val_acc > 0.20:
        print("\n  SUCCESS! HW images CAN be learned with proper training!")
    elif best_val_acc > 0.15:
        print("\n  MARGINAL - some learning happening, might need more epochs/data")
    else:
        print("\n  FAILED - HW images may be too sparse/similar for this model")
        print("  Consider: larger blur kernel, or using software image generation")


if __name__ == "__main__":
    main()
