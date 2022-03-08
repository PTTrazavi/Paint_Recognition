import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model as KerasModel


class Lightweight_model(layers.Layer):
    def __init__(self):
        super().__init__()
        self.features = tf.keras.models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), use_bias=False),
            layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), use_bias=False),
            layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), use_bias=False),
            layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.AveragePooling2D(),
        ])

    def call(self, x):
        x = self.features(x)
        x = layers.Flatten()(x)
        return x


def customized_Lightweight_model(
    input_shape,
):
    x = x_input = layers.Input(input_shape, name='input_image')
    # x = layers.Rescaling(1./255)(x)
    x = Lightweight_model()(x)
    x = layers.Dense(64,activation='relu')(x)
    x = layers.Dense(5,activation='softmax')(x)

    return KerasModel(inputs=x_input, outputs=x)
