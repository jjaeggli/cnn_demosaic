# Defines models and model creation.
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def residual_block(x, filters, kernel_size=3, strides=1, padding="same", activation="relu"):
    """Creates a residual block."""
    residual = x
    x = layers.Conv2D(filters, kernel_size, strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    return x


def create_model(input_shape, output_channels=3, filters=64, residual_blocks=32):
    """Creates a generic ESRGAN-like model with residual blocks."""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(inputs)
    x = layers.Activation("relu")(x)

    # Build residual blocks.
    for _ in range(residual_blocks):
        x = residual_block(x, filters)

    # Additional convolution.
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Long residual connection.
    x = layers.Add()([x, layers.Conv2D(filters, kernel_size=1, padding="same")(inputs)])

    # As this is not an upscaling model, the output layer shape is the same as the input layer, and
    # there is no upscaling block.

    outputs = layers.Conv2D(
        output_channels, kernel_size=3, strides=1, padding="same", activation="tanh"
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def create_32_64_32_model(weights_path=None):
    """Creates a model with 32x32 tile size, 64 filters, and 32 residual blocks. Bayer default."""
    # A single value channel input for use with a raw array.
    input_shape = (32, 32, 1)
    filters = 64
    residual_blocks = 32

    residual_model = create_model(input_shape, filters=filters, residual_blocks=residual_blocks)
    residual_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="mse",
        metrics=["accuracy"],
    )

    if weights_path is not None:
        residual_model.load_weights(weights_path)

    return residual_model


def create_xtrans_model(weights_path=None):
    """Creates a model with 36x36 tile size, 64 filters, and 32 residual blocks. X-Trans default."""
    # The X-Trans model uses 36x36 blocks, as the X-Trans sensor pattern repeats on a 6x6 basis.
    input_shape = (36, 36, 1)
    filters = 64
    residual_blocks = 32

    residual_model = create_model(input_shape, filters=filters, residual_blocks=residual_blocks)
    residual_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="mse",
        metrics=["accuracy"],
    )

    if weights_path is not None:
        residual_model.load_weights(weights_path)

    return residual_model


class FakeModel:
    """Fake model which produces random output for testing."""

    def __init__(self, output_shape=(32, 32, 3)):
        self.output_shape = output_shape

    def predict(self, tiles):
        output_array = np.asarray([np.random.random(self.output_shape) for _ in tiles])
        return output_array


def create_fake_xtrans_model(weights_path=None):
    """This creates a fake model for testing the Processor."""
    return FakeModel(output_shape=(36, 36, 3))
