import tensorflow as tf
from tensorflow.keras import layers, models


def _fire_module(x, squeeze_channels, expand_channels):
    squeeze = layers.Conv2D(
        squeeze_channels, (1, 1), activation='relu', padding='valid',
        kernel_initializer='he_normal'
    )(x)
    expand_1x1 = layers.Conv2D(
        expand_channels, (1, 1), activation='relu', padding='valid',
        kernel_initializer='he_normal'
    )(squeeze)
    expand_3x3 = layers.Conv2D(
        expand_channels, (3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal'
    )(squeeze)
    return layers.Concatenate(axis=-1)([expand_1x1, expand_3x3])


def build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=24, dropout_rate=0.5):
    inputs = layers.Input(shape=input_shape, dtype='float32')
    x = layers.Rescaling(1.0 / 255.0, name='rescale_0_1')(inputs)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1', kernel_initializer='he_normal')(x)
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
    x = layers.Conv2D(num_classes, (1, 1), activation=None, padding='valid', name='conv_final', dtype='float32', kernel_initializer='he_normal')(x)
    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    outputs = layers.Softmax(name='predictions')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='squeezenet_v1_1')
    return model


if __name__ == '__main__':
    print("Building SqueezeNet (common)...")
    m = build_squeezenet_v11(input_shape=(224, 224, 3), num_classes=8)
    m.summary()