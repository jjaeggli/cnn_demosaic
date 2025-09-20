# Module for correcting color and whitebalance from the raw image.

import numpy as np
import tensorflow as tf

from cnn_demosaic.profile import profile
from cnn_demosaic.types import MonochromeParameters


class Color:
    def __init__(self, model):
        self.model = model

    @profile()
    def process(self, img_arr, wb_matrix):
        orig_shape = img_arr.shape
        # Compute the histogram.
        img_linear_arr = None
        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D RGB array.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_linear_arr = tf.reshape(img_arr, new_shape)
        elif len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            img_linear_arr = img_arr

        wb_matrix_full = np.full_like(img_linear_arr, wb_matrix)
        process_arr = np.concatenate((img_linear_arr, wb_matrix_full), axis=1)
        output_arr = self.model.predict(process_arr, batch_size=8192, verbose=0)
        return tf.reshape(output_arr, orig_shape)


class WhiteBalance:
    def __init__(self, model):
        self.model = model

    @profile()
    def process(self, img_arr, wb_matrix):
        orig_shape = img_arr.shape
        img_linear_arr = None

        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D (y, x, 3) RGB array.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_linear_arr = tf.reshape(img_arr, new_shape)
        elif len(img_arr.shape) == 2 and img_arr.shape[-1] == 3:
            # The input shape is a linear (n, 3) RGB array.
            img_linear_arr = img_arr
        else:
            raise ValueError(f"Invalid input image shape: {orig_shape}")

        wb_matrix = wb_matrix.reshape((1, 3))

        params_add, params_mult = self.model.predict(wb_matrix, verbose=0)

        output_arr = (img_linear_arr + params_add) * params_mult

        # Do not reshape the array if the original input is a linear array.
        if len(orig_shape) == 2:
            return output_arr
        return tf.reshape(output_arr, orig_shape)


class ColorTransform:
    def __init__(self, model):
        self.model = model

    @profile()
    def process(self, img_arr):
        orig_shape = img_arr.shape
        img_linear_arr = None

        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D RGB array.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_linear_arr = tf.reshape(img_arr, new_shape)
        elif len(img_arr.shape) == 2 and img_arr.shape[-1] == 3:
            img_linear_arr = img_arr
        else:
            raise ValueError(f"Invalid input image shape: {orig_shape}")

        output_arr = self.model.predict(img_linear_arr, batch_size=8192, verbose=0)

        # Do not reshape the array if the original input is a linear array.
        if len(orig_shape) == 2:
            return output_arr
        return tf.reshape(output_arr, orig_shape)


class MonochromeTransform:
    def __init__(self, params: MonochromeParameters):
        """Initializes the MonochromeTransform.
        Args:
            params (MonochromeParameters): A tuple containing per-channel weights.
                             These weights are used to mix the RGB channels into a single monochrome channel.
        """
        self.weights = (params.ch_r, params.ch_g, params.ch_b)

    @profile()
    def process(self, img_arr):
        # Mix channels based on self.weigths and return a monochrome image array.
        orig_shape = img_arr.shape
        img_linear_arr = None

        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D RGB array (H, W, 3).
            # Reshape to a linear (N, 3) array where N = H * W.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_linear_arr = tf.reshape(img_arr, new_shape)
        elif len(img_arr.shape) == 2 and img_arr.shape[-1] == 3:
            # The input shape is already a linear (N, 3) RGB array.
            img_linear_arr = img_arr
        else:
            raise ValueError(f"Invalid input image shape: {orig_shape}. Expected (H, W, 3) or (N, 3).")

        # Convert weights to a TensorFlow tensor for element-wise multiplication.
        # Ensure the dtype matches the input image data type.
        weights_tensor = tf.constant(self.weights, dtype=img_linear_arr.dtype)

        # Perform the weighted sum across the color channels (axis=-1).
        # This operation will convert the (N, 3) RGB array into an (N,) monochrome array.
        monochrome_linear_arr = tf.reduce_sum(img_linear_arr * weights_tensor, axis=-1)

        # Reshape the output if the original input was a 2D RGB image (H, W, 3).
        # The monochrome output should then be a 2D (H, W) array.
        if len(orig_shape) == 3:
            target_output_shape = (orig_shape[0], orig_shape[1])
            return tf.reshape(monochrome_linear_arr, target_output_shape)

        # If the original input was a linear (N, 3) array, return the (N,) monochrome array directly.
        return monochrome_linear_arr
