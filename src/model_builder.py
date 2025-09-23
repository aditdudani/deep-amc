import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3

def build_googlenet_transfer(input_shape=(224, 224, 3), num_classes=24):
    """
    Builds a GoogLeNet (InceptionV3) model using transfer learning.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes (modulations).

    Returns:
        keras.Model: The compiled Keras model.
    """
    # 1. Load the pre-trained InceptionV3 model without its top classification layer.
    #    'weights="imagenet"' downloads weights pre-trained on the ImageNet dataset.
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # 2. Freeze the layers of the base model. This prevents their weights from being
    #    updated during training, so we only train our new classification head.
    base_model.trainable = False

    # 3. Create our new classification head.
    #    We take the output of the base model and add our own layers.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Flattens the feature maps
    x = Dropout(0.5)(x)              # Regularization to prevent overfitting

    # The final output layer. It has 'num_classes' neurons, one for each modulation type.
    # We use a linear activation because we will use a loss function that expects raw logits.
    predictions = Dense(num_classes, activation=None)(x)

    # 4. Assemble the final model.
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

if __name__ == '__main__':
    # A simple test to verify the model builds correctly
    print("Building model for verification...")
    model = build_googlenet_transfer()
    model.summary()
    print("\nModel built successfully!")