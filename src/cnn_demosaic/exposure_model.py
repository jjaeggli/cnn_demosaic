import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

HIST_SIZE = 32
DENSE_LAYERS = 16


class LogNormalizationLayer(layers.Layer):
    def __init__(self, axis=1, epsilon=1e-6, **kwargs):
        super(LogNormalizationLayer, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        self.input_shape = input_shape

    def call(self, inputs):
        log_inputs = tf.math.log(tf.cast(inputs, dtype=tf.float32) + self.epsilon)

        max_log_input = tf.math.reduce_max(log_inputs, axis=self.axis, keepdims=True)
        normalized_output = log_inputs / (max_log_input + self.epsilon)

        return normalized_output

    def get_config(self):
        config = super(LogNormalizationLayer, self).get_config()
        config.update(
            {
                "axis": self.axis,
                "epsilon": self.epsilon,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def exposure_params_model(hist_size=HIST_SIZE, dense_layers=DENSE_LAYERS):
    """A model which generates processing parameters based on a histogram."""

    y_hist = keras.Input(shape=(hist_size,), name="y_hist")

    # Normalize the input histogram.
    weights_layers = LogNormalizationLayer()(y_hist)

    for i in range(dense_layers):
        weights_layers = layers.Dense(hist_size, activation="relu")(weights_layers)

    # This prevents the output from blowing up.
    weights_layers = layers.Dense(hist_size, activation="sigmoid")(weights_layers)

    levels_weights = layers.Dense(8)(weights_layers)
    levels_weights = layers.Dense(4)(levels_weights)
    levels_weights = layers.Dense(2)(levels_weights)

    gamma_weights = layers.Dense(8)(weights_layers)
    gamma_weights = layers.Dense(4)(gamma_weights)
    gamma_weights = layers.Dense(1)(gamma_weights)

    curve_weights = layers.Dense(8)(weights_layers)
    curve_weights = layers.Dense(4)(curve_weights)
    curve_weights = layers.Dense(3)(curve_weights)

    model = keras.Model(inputs=[y_hist], outputs=[levels_weights, gamma_weights, curve_weights])

    return model


def create_exposure_model(weights_path=None):
    processing_model = exposure_params_model()
    processing_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=["accuracy"],
    )

    if weights_path is not None:
        processing_model.load_weights(weights_path)

    return processing_model
