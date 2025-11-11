import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3

def build_googlenet_transfer(input_shape=(224, 224, 3), num_classes=24, train_base=False):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = bool(train_base)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation=None, dtype='float32')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__ == '__main__':
    print("Building model (common)...")
    m = build_googlenet_transfer()
    m.summary()