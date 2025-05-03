# Module for correcting exposure / luminance signal in the raw image.

import numpy as np
import tensorflow as tf


class Color:
    def __init__(self, model, hist_size=32):
        self.model = model
        self.hist_size = hist_size

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
        output_arr = self.model.predict(process_arr, batch_size=8192)
        return tf.reshape(output_arr, orig_shape)
