# Module for correcting color and whitebalance from the raw image.

import numpy as np
import tensorflow as tf

from cnn_demosaic.profile import profile


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

        # This is probably unnecessary for these operations.
        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            # The input shape is assumed to be a 2D RGB array.
            new_shape = (img_arr.shape[0] * img_arr.shape[1], 3)
            img_linear_arr = tf.reshape(img_arr, new_shape)
        elif len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            img_linear_arr = img_arr
        else:
            raise ValueError(f"Invalid input image shape: {orig_shape}")

        wb_matrix = wb_matrix.reshape((1, 3))

        params_add, params_mult = self.model.predict(wb_matrix, verbose=0)

        output_arr = (img_linear_arr + params_add) * params_mult

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
        elif len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
            img_linear_arr = img_arr
        else:
            raise ValueError(f"Invalid input image shape: {orig_shape}")

        output_arr = self.model.predict(img_linear_arr, batch_size=8192, verbose=0)

        return tf.reshape(output_arr, orig_shape)
