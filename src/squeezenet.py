import tensorflow as tf
from tensorflow.keras import layers, models


def _fire_module(x, squeeze_channels, expand_channels):
    """
    SqueezeNet Fire module: squeeze (1x1) -> expand (1x1 + 3x3) -> concat.
    Args:
        x: input tensor
        squeeze_channels: number of filters in squeeze 1x1 conv
        expand_channels: number of filters in each expand branch (1x1 and 3x3)
    Returns:
        Tensor after concatenation of expand branches
    """
    squeeze = layers.Conv2D(squeeze_channels, (1, 1), activation='relu', padding='valid')(x)
    expand_1x1 = layers.Conv2D(expand_channels, (1, 1), activation='relu', padding='valid')(squeeze)
    expand_3x3 = layers.Conv2D(expand_channels, (3, 3), activation='relu', padding='same')(squeeze)
    return layers.Concatenate(axis=-1)([expand_1x1, expand_3x3])


def build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=24, dropout_rate=0.5):
    """
    Build SqueezeNet v1.1 in Keras with logits output (no softmax).

    Notes:
    - Includes a Rescaling(1/255) layer at the input so you can feed 0..255 images.
    - Final layer produces raw logits (activation=None) for use with from_logits=True losses.
    - Architecture loosely follows SqueezeNet v1.1 configuration.
    """
    inputs = layers.Input(shape=input_shape, dtype='float32')
    x = layers.Rescaling(1.0 / 255.0, name='rescale_0_1')(inputs)

    # v1.1 uses a smaller initial conv
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    x = _fire_module(x, squeeze_channels=16, expand_channels=64)   # fire2
    x = _fire_module(x, squeeze_channels=16, expand_channels=64)   # fire3
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool3')(x)

    x = _fire_module(x, squeeze_channels=32, expand_channels=128)  # fire4
    x = _fire_module(x, squeeze_channels=32, expand_channels=128)  # fire5
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool5')(x)

    x = _fire_module(x, squeeze_channels=48, expand_channels=192)  # fire6
    x = _fire_module(x, squeeze_channels=48, expand_channels=192)  # fire7
    x = _fire_module(x, squeeze_channels=64, expand_channels=256)  # fire8
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool8')(x)
    x = _fire_module(x, squeeze_channels=64, expand_channels=256)  # fire9

    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Final conv to num_classes, then global average pool -> logits
    x = layers.Conv2D(num_classes, (1, 1), activation=None, padding='valid', name='conv_final', dtype='float32')(x)
    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    # Output is logits in float32
    outputs = layers.Activation('linear', dtype='float32', name='logits')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='squeezenet_v1_1')
    return model


if __name__ == '__main__':
    print("Building SqueezeNet v1.1 for verification...")
    m = build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=8)
    m.summary()
    print("\nSqueezeNet v1.1 built successfully.")
