# src/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_googlenet_transfer(input_shape, num_classes):
    """
    Builds a GoogLeNet (InceptionV3) model for transfer learning.

    Args:
        input_shape (tuple): The shape of the input images (e.g., (224, 224, 3)).
        num_classes (int): The number of output classes for the new classifier.

    Returns:
        tf.keras.Model: The compiled Keras model ready for training.
    """
    # Load the InceptionV3 model, pre-trained on the ImageNet dataset.
    # We use `include_top=False` to exclude the original 1000-class classifier.
    base_model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the convolutional base. This prevents the weights of the pre-trained
    # layers from being updated during our training. They will act as a fixed
    # feature extractor.
    base_model.trainable = False
    
    # Create our new classification head on top of the frozen base.
    inputs = layers.Input(shape=input_shape)
    # The base model outputs a feature map.
    x = base_model(inputs, training=False)
    # We then apply Global Average Pooling to reduce the spatial dimensions.
    x = layers.GlobalAveragePooling2D()(x)
    # A Dropout layer is added for regularization to prevent overfitting.
    x = layers.Dropout(0.2)(x)
    # Finally, the Dense output layer has one neuron for each of our classes.
    # It has no activation function because we will use `from_logits=True` in the loss.
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs, outputs)
    return model