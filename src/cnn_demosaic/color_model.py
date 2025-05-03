import numpy as np
import tensorflow as tf

from cnn_demosaic import transform
from tensorflow import keras
from tensorflow.keras import layers


srgb_to_xyz = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

# This roughly matches the sRGB D65 matrix shown here:
# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
xyz_to_srgb = np.linalg.inv(srgb_to_xyz)


class XyzToSrgbLayer(layers.Layer):
    def __init__(self, color_ch=3):
        super(XyzToSrgbLayer, self).__init__()
        self.color_ch = color_ch

    def build(self, input_shape):
        return

    def call(self, inputs):
        return tf.tensordot(inputs, xyz_to_srgb, 1)


class ColorTransformLayer(layers.Layer):
    def __init__(self, color_ch=3):
        super(ColorTransformLayer, self).__init__()
        self.color_ch = color_ch

    def build(self, input_shape):
        self.w = self.add_weight(shape=(3, 3), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.tensordot(inputs, self.w, 1)


class MultiColorTransformLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(MultiColorTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(3, 3), initializer="random_normal", trainable=True)

    def call(self, inputs):
        @tf.function()
        def apply_fn(inputs):
            return tf.tensordot(inputs, self.w, 1)

        return tf.map_fn(
            apply_fn,
            inputs
        )

    def compute_output_shape(self, input_shape):
        return input_shape


# Applies a channel independent s-curve to the image.
class ColorCurveAdjLayer(layers.Layer):
    def __init__(self, color_ch=3):
        super(ColorCurveAdjLayer, self).__init__()
        self.color_ch = color_ch

    def build(self, input_shape):
        self.input_shape = input_shape
        # Using (3,3) weights applies per-channel weights.
        self.w = self.add_weight(shape=(3, 3), initializer="random_normal", trainable=True)

    def call(self, inputs):
        @tf.function()
        def apply_curve(rgb_arr):
            output = transform.tf_s_curve_fn(rgb_arr, self.w[0], self.w[1], self.w[2])
            return output

        return apply_curve(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def wb_model():
    wb_input = keras.Input(shape=(3,), name="wb_input")

    # Apply a set of weights to the white balance. The first white balance weights
    # will be used on the camera rgb input. The second will be used on the
    # converted RGB input.
    wb_transform = layers.Dense(6)(wb_input)
    wb_transform = layers.Dense(6)(wb_transform)
    wb_transform = layers.Dense(6)(wb_transform)

    wb_transform_0 = layers.Dense(6)(wb_transform)
    wb_transform_0 = layers.Dense(6)(wb_transform_0)
    wb_transform_0 = layers.Dense(3)(wb_transform_0)
    wb_transform_0 = ColorTransformLayer()(wb_transform_0)

    wb_transform_1 = layers.Dense(6)(wb_transform)
    wb_transform_1 = layers.Dense(6)(wb_transform_1)
    wb_transform_1 = layers.Dense(3, activation="sigmoid")(wb_transform_1)
    wb_transform_1 = ColorTransformLayer()(wb_transform_1)

    return keras.Model(inputs=wb_input, outputs=[wb_transform_0, wb_transform_1])


def color_transform_model():
    # Accepts RGB input arrays and applies the color transformation.
    rgb_input = keras.Input(shape=(3,), name="rgb_input")

    rgb_layers = layers.Identity()(rgb_input)

    # Apply per channel color curves to the input.
    rgb_layers = ColorCurveAdjLayer()(rgb_layers)

    # Apply the color transformation to the RGB image.
    rgb_layers = ColorTransformLayer()(rgb_layers)
    rgb_layers = XyzToSrgbLayer()(rgb_layers)

    # Apply a second RGB color curve to the input.
    rgb_layers = ColorCurveAdjLayer()(rgb_layers)

    return keras.Model(inputs=rgb_input, outputs=rgb_layers)


def create_composite_model():
    # Merged input contains RGB values for the pixel as [0:3] and white balance
    # values for the pixel as [3:]. The purpose of this approach is to create
    # a general model which can color correct for images of any light source,
    # however it doesn't seem to be working as intended, even though it is
    # doing *something*.
    merged_input = keras.Input(shape=(6,), name="merged_input")
    rgb_input = layers.Lambda(lambda x: x[:, 0:3])(merged_input)
    wb_input = layers.Lambda(lambda x: x[:, 3:])(merged_input)

    wb_transform_0, wb_transform_1 = wb_model()(wb_input)

    rgb_layers = layers.Identity()(rgb_input)

    # Add the transformed white balance signal.
    rgb_layers = keras.layers.Add()([rgb_layers, wb_transform_0])

    # Apply a transformed white balance.
    rgb_layers = layers.Multiply()([rgb_layers, wb_transform_1])

    rgb_layers = color_transform_model()(rgb_layers)

    model = keras.Model(inputs=merged_input, outputs=rgb_layers)

    return model


def create_color_model():
    # Merged input contains RGB values for the pixel as [0:3] and white balance
    # values for the pixel as [3:]. The purpose of this approach is to create
    # a general model which can color correct for images of any light source,
    # however it doesn't seem to be working as intended, even though it is
    # doing *something*.
    merged_input = keras.Input(shape=(6,), name="merged_input")
    rgb_input = layers.Lambda(lambda x: x[:, 0:3])(merged_input)
    wb_input = layers.Lambda(lambda x: x[:, 3:])(merged_input)

    # Apply a set of weights to the white balance. The first white balance weights
    # will be used on the camera rgb input. The second will be used on the
    # converted RGB input.
    wb_transform = layers.Dense(6)(wb_input)
    wb_transform = layers.Dense(6)(wb_transform)
    wb_transform = layers.Dense(6)(wb_transform)

    wb_transform_1 = layers.Dense(3)(wb_transform)
    wb_transform_1 = layers.Dense(3, activation="sigmoid")(wb_transform_1)
    wb_transform_1 = ColorTransformLayer()(wb_transform_1)

    wb_transform_2 = layers.Dense(6)(wb_transform)
    wb_transform_2 = layers.Dense(6)(wb_transform_2)
    wb_transform_2 = layers.Dense(3)(wb_transform_2)
    wb_transform_2 = layers.Dense(3, activation="sigmoid")(wb_transform_2)
    wb_transform_2 = ColorTransformLayer()(wb_transform_2)

    rgb_layers = layers.Identity()(rgb_input)

    # Apply a transformed white balance.
    rgb_layers = layers.Multiply()([rgb_layers, wb_transform_1])

    # Apply per channel color curves to the input.
    rgb_layers = ColorCurveAdjLayer()(rgb_layers)

    # Apply the color transformation to the RGB image.
    rgb_layers = ColorTransformLayer()(rgb_layers)
    rgb_layers = XyzToSrgbLayer()(rgb_layers)

    # Apply a second RGB color curve to the input.
    rgb_layers = ColorCurveAdjLayer()(rgb_layers)

    # Apply a post RGB conversion white balance layer. Performance is better with this enabled.
    # One wonders what this is actually doing!!
    rgb_layers = layers.Multiply()([rgb_layers, wb_transform_2])

    model = keras.Model(inputs=merged_input, outputs=rgb_layers)

    return model


def build_color_model(weights_path=None):
    processing_model = create_color_model()
    processing_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=["accuracy"],
    )

    if weights_path is not None:
        processing_model.load_weights(weights_path)

    return processing_model
